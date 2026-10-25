"""Frozen final evaluation, resumable predictions, common G4 latency benchmark.

Call chuan_bi_kiem_tra() AFTER training. Inspect GT previews/protocol, then
chay_kiem_tra(). Never selects epochs, tunes thresholds, or trains on TEST.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import csv
import gc
import io
import json
import time
import uuid
import numpy as np
from PIL import Image, ImageDraw
import quy_trinh_g4_common as C
import danh_gia_voc_core as M

OUTPUT = C.PROJECT/'evaluations/final_g4_v1'
NAMES = ('YOLOv8s_G4', 'YOLOv8s_BTA_G4', 'Faster_RCNN_R50_FPN', 'SSD300_VGG16')
OWN_SOURCES = ('quy_trinh_g4_common.py', 'huan_luyen_cap_yolo_g4.py',
               'danh_gia_voc_core.py', 'danh_gia_4_mo_hinh_g4.py')


def sources():
    return {name: C.sha_file(C.ROOT/name) for name in (*C.PINS, *OWN_SOURCES)}


def freeze_models(torch, env):
    C.require_complete(torch, C.PROJECT, C.RUNS)
    from huan_luyen_cap_yolo_g4 import normalize_pair
    configs = [json.loads((C.PROJECT/'formal_runs'/n/'config.json').read_bytes()) for n in C.RUNS]
    assert normalize_pair(configs[0]) == normalize_pair(configs[1])
    assert configs[0]['actual_training_gpu_required'] == configs[1]['actual_training_gpu_required'] == C.G4
    helper = C.pinned('huan_luyen_thu_yolov8s_voc2007.py')
    _, metadata = helper.read_final_split(C.PROJECT)
    assert all(c['classes'] == list(helper.CLASSES) and c['seed'] == 42 for c in configs)
    records = []
    for name, folder_name, config in zip(NAMES, C.RUNS, configs):
        folder = C.PROJECT/'formal_runs'/folder_name
        best = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        selection = 'EMA' if name.startswith('YOLO') else 'raw'
        assert best['classes'] == list(helper.CLASSES)
        expected = ('formal_baseline_best_ema' if selection == 'EMA' else
                    'formal_fasterrcnn_best_raw' if name.startswith('Faster') else 'formal_ssd300_best_raw')
        assert best['purpose'] == expected
        record = dict(name=name, run=folder_name, completed_epochs=100,
                      best_epoch=best['epoch'], best_validation_map50_voc07=best['map50_voc07'],
                      best_sha256=C.sha_file(folder/'best.pt'), config_sha256=C.digest(C.jb(config)),
                      history_sha256=C.sha_file(folder/'history.csv'), selection=selection,
                      training_hardware=('G4 only, new full run' if selection == 'EMA' else
                          'epochs 1-15 Tesla T4; epochs 16-100 G4' if name.startswith('Faster') else 'G4'),
                      note='Training history retained honestly; latency will be measured on the same actual G4.')
        del best
        records.append(record)
    return dict(version='final_four_G4_v1', classes=list(helper.CLASSES), models=records,
                split_sha256=metadata['sha256'], source_sha256=sources(),
                inference_software={k: env[k] for k in ('numpy', 'pillow', 'torch', 'torchvision',
                    'torch_cuda', 'ultralytics', 'opencv_imported', 'gpu')},
                protocol=M.PROTOCOL, test_used_for_selection=False,
                note='Three detector architectures, four configurations; recipes/backbones/resolutions differ.')


def read_image(root, row):
    with Image.open(root/'JPEGImages'/f"{row['source_id']}.jpg") as image:
        rgb = image.convert('RGB')
    if row['kind'] == 'crop':
        rgb = rgb.crop(tuple(row['window']))
    assert rgb.size == (row['annotation']['width'], row['annotation']['height'])
    return rgb


def gt_preview(root, rows, path):
    selected = []
    for level in ('original', *M.LEVELS):
        selected.extend([r for r in rows if (r['kind'] == 'original' if level == 'original'
                         else r.get('requested_level') == level)][:2])
    canvas = Image.new('RGB', (800, 250*len(selected)), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected):
        image = read_image(root, row)
        scale = min(500/image.width, 220/image.height)
        resized = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))))
        x, y = 8, index*250+24
        canvas.paste(resized, (x, y))
        draw.text((8, index*250+5), row['view_id'], fill='black')
        for obj in row['annotation']['objects']:
            box = obj['box']
            color = 'orange' if obj.get('benchmark_ignore') or obj['difficult'] else 'green'
            draw.rectangle((x+box[0]*scale, y+box[1]*scale, x+box[2]*scale, y+box[3]*scale),
                           outline=color, width=2)
            draw.text((x+box[0]*scale, y+box[1]*scale), str(obj['class_name']), fill=color)
        draw.text((525, index*250+30), 'Green = eligible GT\nOrange = ignored GT\nNo predictions here.', fill='black')
    canvas.save(path)


def prepare_manifest(root, helper, geometry, test_ids):
    path = OUTPUT/'views.jsonl'
    if path.exists():
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        audit = json.loads((OUTPUT/'dataset_audit.json').read_bytes())
        assert C.sha_file(path) == audit['views_sha256']
        assert len(rows) == audit['total_views'] and len({r['view_id'] for r in rows}) == len(rows)
        for index, row in enumerate([r for r in rows if r['kind'] == 'original'], 1):
            assert C.sha_file(root/'JPEGImages'/f"{row['source_id']}.jpg") == row['image_sha256']
            assert C.sha_file(root/'Annotations'/f"{row['source_id']}.xml") == row['xml_sha256']
            if index % 1000 == 0:
                print(f'Kiem tra anh TEST da dong bang: {index}/4952', flush=True)
        if not (OUTPUT/'kiem_tra_nhan_TEST_goc_va_cat.png').exists():
            gt_preview(root, rows, OUTPUT/'kiem_tra_nhan_TEST_goc_va_cat.png')
        return rows
    rows, failures = [], []
    for index, image_id in enumerate(test_ids, 1):
        ann = helper.read_annotation(root/'Annotations'/f'{image_id}.xml')
        with Image.open(root/'JPEGImages'/f'{image_id}.jpg') as image:
            assert image.size == (ann['width'], ann['height'])
        row = dict(view_id=image_id, source_id=image_id, kind='original', annotation=ann,
                   image_sha256=C.sha_file(root/'JPEGImages'/f'{image_id}.jpg'),
                   xml_sha256=C.sha_file(root/'Annotations'/f'{image_id}.xml'))
        rows.append(row)
        derived, failed = M.make_views(image_id, ann, geometry)
        rows.extend(derived); failures.extend(failed)
        if index % 500 == 0 or index == len(test_ids):
            print(f'Tao danh sach TEST goc/cat: {index}/{len(test_ids)}', flush=True)
    content = ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n' for r in rows).encode('utf-8')
    counts = Counter((r.get('requested_level', 'original'), r.get('edge', 'none')) for r in rows)
    audit = dict(original_images=len(test_ids), total_views=len(rows),
                 derived_views=len(rows)-len(test_ids), views_sha256=C.digest(content),
                 counts_by_requested_level_and_edge={str(k): v for k, v in sorted(counts.items())},
                 failures_by_reason=dict(Counter(f['reason'] for f in failures)),
                 failed_views=failures,
                 eligible_crop_objects_by_level=dict(Counter(o['level'] for r in rows if r['kind'] == 'crop'
                     for o in r['annotation']['objects'] if not o['benchmark_ignore'])),
                 ignored_crop_objects=sum(o['benchmark_ignore'] for r in rows if r['kind'] == 'crop'
                     for o in r['annotation']['objects']),
                 note='Controlled targets exclude natural truncation. r is bbox area retained, not amodal visibility.')
    # Audit committed first; an interrupted manifest commit can be retried without changing its bytes.
    C.once(OUTPUT/'dataset_audit.json', C.jb(audit))
    C.atomic(path, content)
    gt_preview(root, rows, OUTPUT/'kiem_tra_nhan_TEST_goc_va_cat.png')
    return rows


def chuan_bi_kiem_tra():
    """Freeze all four selected checkpoints and fixed TEST crops, NO inference."""
    torch, _, env = C.prepare_runtime()
    with C.job_lock():
        frozen = freeze_models(torch, env)
        before = C.snapshot(include_new=True)
        try:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            C.once(OUTPUT/'models_frozen.json', C.jb(frozen))
            helper, geometry = C.pinned('huan_luyen_thu_yolov8s_voc2007.py'), C.pinned('kiem_tra_bta_cpu_voc2007.py')
            ids, _ = helper.read_final_split(C.PROJECT)
            root = C.restore_voc(helper, ('test',))['test']
            rows = prepare_manifest(root, helper, geometry, ids['test'])
            print('DA DONG BANG 4 BEST + TEST: ' + str(len(rows)) + ' view.', flush=True)
            print('Xem kiem_tra_nhan_TEST_goc_va_cat.png va HUONG_DAN. Chua chay du doan.', flush=True)
            return str(OUTPUT)
        finally:
            assert C.snapshot(include_new=True) == before


def build_detector(index, torch, torchvision, frozen):
    record = frozen['models'][index]
    run = C.PROJECT/'formal_runs'/record['run']
    assert C.sha_file(run/'best.pt') == record['best_sha256']
    best = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    config = json.loads((run/'config.json').read_bytes())
    assert best['config_sha256'] == record['config_sha256'] == C.digest(C.jb(config))
    if index < 2:
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
        from types import SimpleNamespace
        pretrain = C.PROJECT/'checkpoints/pretrained/yolov8s.pt'
        assert C.sha_file(pretrain) == config['pretrained_sha256']
        base = YOLO(str(pretrain))
        model = DetectionModel(deepcopy(base.model.yaml), nc=20, verbose=False)
        model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
        model.names, model.nc = dict(enumerate(frozen['classes'])), 20
        del base
    elif index == 2:
        pilot = C.pinned('huan_luyen_thu_fasterrcnn_voc2007_v2.py')
        assert (C.PROJECT/'checkpoints/initialization/fasterrcnn_r50fpn_voc20_seed42_init.pt').is_file()
        model, *_ = pilot.build_model_and_initialization(torch, torchvision, C.PROJECT, frozen['classes'])
    else:
        ssd = C.pinned('huan_luyen_ssd300_voc2007.py')
        assert (C.PROJECT/'checkpoints/initialization/ssd300_vgg16_voc20_seed42_init.pt').is_file()
        model, *_ = ssd.build_model(torch, torchvision, C.PROJECT, frozen['classes'])
    model.load_state_dict(best['model_state_dict'], strict=True)
    del best
    model.eval().float().to('cuda')
    return model


def infer(model, index, images, torch, geometry, engine, pilot):
    """Common scope: in-memory PIL RGB -> preprocessing -> detector/NMS -> CPU boxes."""
    if index < 2:
        from ultralytics.utils.nms import non_max_suppression
        arrays, transforms = [], []
        for image in images:
            w, h = image.size
            ann = dict(width=w, height=h, objects=[])
            arrays.append(geometry.letterbox_rgb(image, ann)['img'])
            gain = min(640/w, 640/h)
            transforms.append((w, h, gain, (640-max(1, round(w*gain)))//2,
                                (640-max(1, round(h*gain)))//2))
        tensor = torch.from_numpy(np.stack(arrays)).to('cuda').float()/255.
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
            output = model(tensor)
            detected = non_max_suppression(output, conf_thres=.001, iou_thres=.7,
                multi_label=True, max_det=100, nc=20, max_time_img=float('inf'))
        result = [engine.restore_boxes(p.detach().float().cpu().numpy(), transform)
                  for p, transform in zip(detected, transforms)]
    else:
        arrays = [np.asarray(im, dtype=np.uint8).transpose(2, 0, 1).copy() for im in images]
        tensors = [torch.from_numpy(a).to('cuda').float()/255. for a in arrays]
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
            output = model(tensors)
        result = []
        for image, pred in zip(images, output):
            b, s, label = [pred[k].detach().cpu().numpy() for k in ('boxes', 'scores', 'labels')]
            w, h = image.size
            pilot.check_prediction_bounds(b, w, h, 'final_test')
            boxes = b.astype(np.float64, copy=True)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            assert (boxes[:, 2:] >= boxes[:, :2]).all() and ((label >= 1) & (label <= 20)).all()
            result.append(np.column_stack([boxes, s, label-1]))
    assert len(result) == len(images)
    return result


def cached_predictions(model, index, rows, root, torch, geometry, engine, pilot, frozen, maximum_chunks):
    folder = OUTPUT/'predictions'/NAMES[index]
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint_sha256=frozen['models'][index]['best_sha256'],
                    manifest_sha256=C.sha_file(OUTPUT/'views.jsonl'),
                    frozen_sha256=C.digest(C.jb(frozen)), batch_size=16 if index != 2 else 2,
                    precision='FP32; TF32 off', chunk_views=128)
    C.once(folder/'identity.json', C.jb(identity))
    chunks_done = 0
    for chunk_index, begin in enumerate(range(0, len(rows), 128)):
        chunk_rows = rows[begin:begin+128]
        chunk_path = folder/f'chunk_{chunk_index:05d}.json'
        checksum_path = chunk_path.with_suffix('.sha256')
        if chunk_path.exists():
            assert checksum_path.exists() and C.sha_file(chunk_path) == checksum_path.read_text().strip(), \
                'Chunk dang do/sai SHA; KHONG thay bang du doan rong: ' + str(chunk_path)
            data = json.loads(chunk_path.read_bytes())
            assert data['identity_sha256'] == C.digest(C.jb(identity))
            assert [r['view_id'] for r in data['rows']] == [r['view_id'] for r in chunk_rows]
            continue
        if maximum_chunks is not None and chunks_done >= maximum_chunks:
            return False
        detected_rows = []
        batch = identity['batch_size']
        for offset in range(0, len(chunk_rows), batch):
            selected = chunk_rows[offset:offset+batch]
            images = [read_image(root, row) for row in selected]
            output = infer(model, index, images, torch, geometry, engine, pilot)
            for row, pred in zip(selected, output):
                M.validate_predictions({row['view_id']: row['annotation']}, {row['view_id']: pred})
                detected_rows.append(dict(view_id=row['view_id'], detections=pred.tolist()))
        content = C.jb(dict(identity_sha256=C.digest(C.jb(identity)), rows=detected_rows))
        # Sidecar first, data committed last. Interrupted uncommitted chunks are recomputed.
        C.atomic(checksum_path, (C.digest(content)+'\n').encode())
        C.atomic(chunk_path, content)
        chunks_done += 1
        print(f'{NAMES[index]}: {min(begin+128, len(rows))}/{len(rows)} TEST views da luu.', flush=True)
    return True


def load_predictions(index, rows):
    folder = OUTPUT/'predictions'/NAMES[index]
    predictions = {}
    identity_sha = C.digest(C.jb(json.loads((folder/'identity.json').read_bytes())))
    for chunk_index, begin in enumerate(range(0, len(rows), 128)):
        path = folder/f'chunk_{chunk_index:05d}.json'
        assert C.sha_file(path) == path.with_suffix('.sha256').read_text().strip()
        chunk = json.loads(path.read_bytes())
        assert chunk['identity_sha256'] == identity_sha
        assert [r['view_id'] for r in chunk['rows']] == [r['view_id'] for r in rows[begin:begin+128]]
        for item in chunk['rows']:
            assert item['view_id'] not in predictions
            predictions[item['view_id']] = np.asarray(item['detections'], dtype=np.float64).reshape(-1, 6)
    assert set(predictions) == {r['view_id'] for r in rows}
    return predictions


def latency(model, index, rows, root, torch, geometry, engine, pilot, frozen, env, session_folder):
    path = session_folder/f'{NAMES[index]}.json'
    protocol = dict(batch=1, precision='FP32; TF32 off', warmup_calls=20, timed_calls=100,
                    scope='PIL RGB in RAM -> CPU preprocessing + H2D + detector incl NMS + D2H + canonical boxes; synchronized',
                    disk_decode_excluded=True, parameter_count=sum(p.numel() for p in model.parameters()),
                    gpu=C.G4, clock_or_power_locked=False, exclusive_runtime_required=True,
                    no_training_time_as_inference_FPS=True)
    originals = [r for r in rows if r['kind'] == 'original']
    selected = M.keyed_rng('latency', 42).sample(originals, 100)
    protocol['view_ids'] = [r['view_id'] for r in selected]
    protocol['checkpoint_sha256'] = frozen['models'][index]['best_sha256']
    protocol['frozen_sha256'] = C.digest(C.jb(frozen))
    protocol['same_session_benchmark_id'] = session_folder.name
    if path.exists():
        record = json.loads(path.read_bytes())
        assert record['protocol'] == protocol
        return record
    images = [read_image(root, r) for r in selected]
    for i in range(20):
        infer(model, index, [images[i]], torch, geometry, engine, pilot)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for image in images:
        torch.cuda.synchronize()
        started = time.perf_counter()
        infer(model, index, [image], torch, geometry, engine, pilot)
        torch.cuda.synchronize()
        times.append((time.perf_counter()-started)*1000)
    mean = float(np.mean(times))
    result = dict(protocol=protocol, environment=env, samples_ms=times,
                  mean_ms=mean, median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                  sequential_images_per_second=1000/mean,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
                  time_utc=datetime.now(timezone.utc).isoformat(),
                  note='One hundred fixed original images, one pass. Hardware clocks/load not locked; not a universal speed guarantee.')
    C.once(path, C.jb(result))
    return result


def benchmark_all(torch, torchvision, rows, root, geometry, engine, pilot, frozen, env):
    """All four timings in ONE runtime/pass; incomplete timing passes are not mixed."""
    folder = OUTPUT/'latency'
    pointer = folder/'COMPLETE.json'
    if pointer.exists():
        record = json.loads(pointer.read_bytes())
        assert record['frozen_sha256'] == C.digest(C.jb(frozen))
        for name in NAMES:
            path = folder/f'{name}.json'
            assert C.sha_file(path) == record['files_sha256'][name]
            assert json.loads(path.read_bytes())['protocol']['same_session_benchmark_id'] == record['session']
        return
    session = folder/('session_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
    session.mkdir(parents=True, exist_ok=False)
    records = []
    for index in range(4):
        model = None
        try:
            print('Do toc do batch1 FP32, CUNG PHIEN G4: '+NAMES[index], flush=True)
            model = build_detector(index, torch, torchvision, frozen)
            records.append(latency(model, index, rows, root, torch, geometry, engine, pilot, frozen, env, session))
        finally:
            del model
            gc.collect(); torch.cuda.empty_cache()
    hashes = {}
    for name, record in zip(NAMES, records):
        content = C.jb(record)
        C.atomic(folder/f'{name}.json', content)
        hashes[name] = C.digest(content)
    C.atomic(pointer, C.jb(dict(session=session.name, files_sha256=hashes,
        frozen_sha256=C.digest(C.jb(frozen)), environment=env, all_four_measured_in_one_runtime=True)))


def summarize_model(index, rows, engine, frozen):
    destination = OUTPUT/'metrics'/f'{NAMES[index]}.json'
    if destination.exists():
        record = json.loads(destination.read_bytes())
        assert record['frozen_sha256'] == C.digest(C.jb(frozen))
        return record
    predictions = load_predictions(index, rows)
    result = dict(model=NAMES[index], frozen_sha256=C.digest(C.jb(frozen)),
                  selected_on_original_validation=True, test_used_for_training=False)
    for kind in ('original', 'crop'):
        subset = [r for r in rows if r['kind'] == kind]
        assert subset, 'Khong co anh trong nhom ' + kind
        annotations = {r['view_id']: r['annotation'] for r in subset}
        pred = {r['view_id']: predictions[r['view_id']] for r in subset}
        print('Cham CPU: ' + NAMES[index] + ' / ' + kind + ' ...', flush=True)
        result[kind] = M.evaluate_suite(annotations, pred, frozen['classes'], cropped=kind == 'crop', engine=engine)
    C.once(destination, C.jb(result))
    return result


def write_tables_and_figures(rows, frozen):
    results = [json.loads((OUTPUT/'metrics'/f'{n}.json').read_bytes()) for n in NAMES]
    timings = [json.loads((OUTPUT/'latency'/f'{n}.json').read_bytes()) for n in NAMES]
    summary_rows = []
    for name, record, timing in zip(NAMES, results, timings):
        for kind in ('original', 'crop'):
            groups = {'all': record[kind]['map50_voc07']}
            if kind == 'crop':
                groups.update(record[kind]['by_object_level'])
            for group, metric in groups.items():
                summary_rows.append(dict(model=name, test_set=kind, object_group=group,
                    map50_voc07=metric['map'], evaluated_classes=metric['evaluated_classes'],
                    precision_at_025=metric['precision_at_025'], recall_at_025=metric['recall_at_025'],
                    f1_at_025=metric['f1_at_025'], tp=metric['tp'], fp=metric['fp'], fn=metric['fn'],
                    median_latency_ms=timing['median_ms'], mean_latency_ms=timing['mean_ms'],
                    common_G4_sequential_images_per_second=timing['sequential_images_per_second']))
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(summary_rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(summary_rows)
    C.atomic(OUTPUT/'bang_so_sanh_4_cau_hinh.csv', output.getvalue().encode('utf-8-sig'))
    class_rows = []
    for name, record in zip(NAMES, results):
        for kind in ('original', 'crop'):
            groups = {'all': record[kind]['map50_voc07']}
            if kind == 'crop':
                groups.update(record[kind]['by_object_level'])
            for group, metric in groups.items():
                for class_name, values in metric['per_class'].items():
                    class_rows.append(dict(model=name, test_set=kind, object_group=group,
                        class_name=class_name, ap50_voc07=values['ap'], positives=values['positives'],
                        tp=values['tp_at_025'], fp=values['fp_at_025'], fn=values['fn_at_025']))
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(class_rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(class_rows)
    C.atomic(OUTPUT/'bang_ket_qua_tung_lop.csv', output.getvalue().encode('utf-8-sig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ('Original', 'Cropped-all', 'Light', 'Medium', 'Heavy')
    for name, record in zip(NAMES, results):
        values = [record['original']['map50_voc07']['map'], record['crop']['map50_voc07']['map']]
        values += [record['crop']['by_object_level'][s]['map'] for s in M.LEVELS]
        ax.plot(labels, [np.nan if v is None else v*100 for v in values], marker='o', label=name)
    ax.set_ylabel('mAP50 VOC07 (%)'); ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title('Final TEST; inspect eligible-class counts before comparing strata')
    fig.tight_layout(); fig.savefig(OUTPUT/'so_sanh_TEST_goc_va_cat.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, run_name in zip(NAMES, C.RUNS):
        with (C.PROJECT/'formal_runs'/run_name/'history.csv').open(encoding='utf-8-sig', newline='') as stream:
            history = list(csv.DictReader(stream))
        assert len(history) == 100
        ax.plot([int(r['epoch']) for r in history], [float(r['map50_voc07'])*100 for r in history], label=name)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Original VAL mAP50 VOC07 (%)'); ax.grid(alpha=.3)
    ax.legend(fontsize=8); ax.set_title('Validation curves; NOT final test results')
    fig.tight_layout(); fig.savefig(OUTPUT/'duong_cong_xac_thuc_100_epoch.png', dpi=180); plt.close(fig)
    # Fixed source IDs, not selected by which model won. Include misses automatically.
    chosen = [r for r in rows if r['kind'] == 'crop'][:6]
    C.once(OUTPUT/'qualitative_fixed_view_ids.json', C.jb([r['view_id'] for r in chosen]))
    C.atomic(OUTPUT/'FINAL_COMPLETE.json', C.jb(dict(status='FOUR_CONFIGURATION_TEST_COMPLETE',
        frozen_sha256=C.digest(C.jb(frozen)), original_test_images=sum(r['kind'] == 'original' for r in rows),
        total_views=len(rows), models=list(NAMES), latency_device=C.G4,
        inference_only_on_common_G4=True, training_hardware_not_rewritten=True,
        limitations=['single seed', 'different architecture recipes/resolutions/backbones',
                     'controlled bbox truncation is not true semantic visibility',
                     'artificial crops share sources; not independent photographs',
                     'sensitivity studies/private 99-image application evaluation not performed here'])))


def qualitative(root, rows):
    chosen = [r for r in rows if r['kind'] == 'crop'][:6]
    if not chosen:
        return
    predictions = [load_predictions(i, rows) for i in range(4)]
    canvas = Image.new('RGB', (1200, len(chosen)*240), 'white')
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(chosen):
        source = read_image(root, row)
        scale = min(290/source.width, 205/source.height)
        image = source.resize((max(1, round(source.width*scale)), max(1, round(source.height*scale))))
        for i, name in enumerate(NAMES):
            x, y = i*300+3, row_index*240+28
            canvas.paste(image, (x, y))
            draw.text((x, row_index*240+3), name+' '+row['view_id'], fill='black')
            for obj in row['annotation']['objects']:
                b = obj['box']
                draw.rectangle((x+b[0]*scale, y+b[1]*scale, x+b[2]*scale, y+b[3]*scale),
                               outline='orange' if obj.get('benchmark_ignore') else 'green', width=1)
            for p in predictions[i][row['view_id']]:
                if p[4] >= .25:
                    draw.rectangle((x+p[0]*scale, y+p[1]*scale, x+p[2]*scale, y+p[3]*scale), outline='red', width=2)
                    draw.text((x+p[0]*scale, y+p[1]*scale), f'{int(p[5])}:{p[4]:.2f}', fill='red')
    canvas.save(OUTPUT/'vi_du_du_doan_4_cau_hinh.png')


def tong_hop_ket_qua_cpu():
    """May run after GPU work, even on a CPU runtime; no installs/model weights loaded."""
    frozen = json.loads((OUTPUT/'models_frozen.json').read_bytes())
    assert frozen['source_sha256'] == sources()
    assert (OUTPUT/'latency/COMPLETE.json').is_file(), 'Chua xong du doan/benchmark GPU.'
    rows = [json.loads(s) for s in (OUTPUT/'views.jsonl').read_text(encoding='utf-8').splitlines()]
    assert C.sha_file(OUTPUT/'views.jsonl') == json.loads((OUTPUT/'dataset_audit.json').read_bytes())['views_sha256']
    engine = C.pinned('huan_luyen_baseline_voc2007.py')
    for index in range(4):
        summarize_model(index, rows, engine, frozen)
    write_tables_and_figures(rows, frozen)
    print('DA CHAM XONG 4 CAU HINH. Ket qua: ' + str(OUTPUT), flush=True)
    return json.loads((OUTPUT/'FINAL_COMPLETE.json').read_bytes())


def chay_kiem_tra(maximum_chunks_per_model=None, tong_hop=True):
    """No limit => finish all. A positive chunk limit permits short resumable sessions."""
    assert maximum_chunks_per_model is None or (isinstance(maximum_chunks_per_model, int) and maximum_chunks_per_model > 0)
    assert (OUTPUT/'models_frozen.json').is_file(), 'Chay chuan_bi_kiem_tra() truoc; xem anh GT va protocol.'
    torch, torchvision, env = C.prepare_runtime()
    with C.job_lock():
        frozen = json.loads((OUTPUT/'models_frozen.json').read_bytes())
        assert frozen == freeze_models(torch, env), 'Trong so/source/runtime da thay doi sau dong bang; dung de kiem tra.'
        before = C.snapshot(include_new=True)
        try:
            helper = C.pinned('huan_luyen_thu_yolov8s_voc2007.py')
            geometry = C.pinned('kiem_tra_bta_cpu_voc2007.py')
            engine = C.pinned('huan_luyen_baseline_voc2007.py')
            pilot = C.pinned('huan_luyen_thu_fasterrcnn_voc2007_v2.py')
            ids, _ = helper.read_final_split(C.PROJECT)
            root = C.restore_voc(helper, ('test',))['test']
            rows = prepare_manifest(root, helper, geometry, ids['test'])
            done = []
            for index in range(4):
                model = None
                try:
                    model = build_detector(index, torch, torchvision, frozen)
                    complete = cached_predictions(model, index, rows, root, torch, geometry, engine, pilot,
                                                  frozen, maximum_chunks_per_model)
                finally:
                    del model
                    gc.collect(); torch.cuda.empty_cache()
                done.append(complete)
            if not all(done):
                print('DA LUU DOAN DU DOAN. Chay lai CUNG LENH de tiep tuc; chua co ket luan cuoi.', flush=True)
                return {'status': 'PREDICTION_SEGMENT_COMPLETE', 'complete_by_model': dict(zip(NAMES, done))}
            benchmark_all(torch, torchvision, rows, root, geometry, engine, pilot, frozen, env)
            print('PHAN GPU DA XONG. Bat dau tinh chi so CPU tu du doan da luu.', flush=True)
            qualitative(root, rows)
            if tong_hop:
                return tong_hop_ket_qua_cpu()
            print('Co the chay tong_hop_ket_qua_cpu() rieng; chua co ket luan cuoi.', flush=True)
            return {'status': 'GPU_PREDICTIONS_AND_COMMON_BENCHMARK_COMPLETE'}
        finally:
            assert C.snapshot(include_new=True) == before, 'Tep huan luyen thay doi khi cham TEST.'
