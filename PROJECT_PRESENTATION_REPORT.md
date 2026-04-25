# Bao cao chi tiet du an Multi-Person Tracking and Pose-based Action Analysis

Ngay lap bao cao: 2026-04-25

Muc dich tai lieu: dung lam tai lieu nen de tao slide, thuyet trinh do an, demo va tra loi cau hoi ve kien truc, mo hinh, du lieu, runtime va cac van de thuc nghiem cua du an.

Ghi chu ve noi dung: Bao cao nay duoc tong hop tu code hien tai trong repository, gom `README.md`, `TEST_GUIDELINES.md`, `pyqt_app.py`, `src/runtime_shared.py`, `train_professional_v3.py`, cac script xu ly du lieu, cac artifact train trong `runs/`, dataset card trong `config/data/`, va cac output debug trong `runs/qt_outputs/`.

## 1. Tom tat ngan gon

Du an xay dung he thong nhan dien hanh dong nguoi dua tren video. Pipeline chinh gom:

- Phat hien nguoi bang YOLOv8 pose.
- Lay keypoints COCO 17 diem cho tung nguoi.
- Gan va duy tri tracking ID bang ByteTrack hoac BoT-SORT.
- Gom keypoint sequence theo tung tracking ID.
- Dua sequence vao action model de du doan hanh dong.
- Hien thi ket qua tren PyQt6 UI va xuat video/debug timeline.

He thong hien tap trung vao 5 nhan hanh dong:

- `Sitting`
- `Walking`
- `Standing`
- `Fall`
- `Lying_Down`

Trang thai runtime hien tai:

- App chinh: `pyqt_app.py`.
- Runtime dung chung: `src/runtime_shared.py`.
- Action model active: `runs/train_bigru_prod5_best_v1/final_safe_system.pth`.
- Backend action model active trong cac run gan day: `torch`.
- Pose backend co the la `TensorRT` neu co `yolov8n-pose.engine` va moi truong co TensorRT/CUDA, nguoc lai dung `PyTorch` voi `yolov8n-pose.pt`.

Ket luan ky thuat ngan gon:

- Detection va tracking da co nen tang tot, nhung classification hanh dong con phu thuoc manh vao chat luong keypoints, occlusion, tracking ID, montage/cut canh va phan bo du lieu train.
- Validation tren dataset co the cao, nhung video thuc te van sai do domain shift va do runtime postprocess.
- Nen thuyet trinh ro day la he thong real-time pose-based action recognition, khong phai chi la model classify doc lap.

## 2. Bai toan va dong co

Bai toan:

- Tu video dau vao, can phat hien nguoi, theo doi tung nguoi qua thoi gian va nhan dien hanh dong cua moi nguoi.
- Truong hop quan trong nhat la phat hien nga, vi co ung dung trong giam sat an toan, cham soc nguoi gia, camera an ninh va phan tich hanh vi.

Thach thuc:

- Nguoi co the bi che khuat mot phan co the.
- Video co the co nhieu nguoi, nhieu ID, goc quay khac nhau.
- Hanh dong nga co nhieu dang: nga ngang, nga tu ghe, nga ve truoc, nga cham, nga nhanh.
- `Standing`, `Walking`, `Sitting` de bi nham neu chi nhin skeleton trong thoi gian ngan.
- Khi nguoi ra khoi khung hinh, bbox/keypoints thay doi dot ngot co the bi hieu nham la `Fall`.
- Video montage ghep nhieu canh nhanh lam tracking ID va label history bi nhiem nhan cu.

## 3. Muc tieu cua du an

Muc tieu chinh:

- Tao pipeline nhan dien hanh dong nhieu nguoi tren video.
- Hien thi truc quan tren UI PyQt6.
- Ho tro test nhanh cac che do runtime: Fast, Balanced, Quality.
- Luu debug timeline de phan tich nguyen nhan sai.
- Co kha nang train/retrain action model tu du lieu pose sequence.

Muc tieu thuc te hien tai:

- Uu tien 5 class: `Sitting`, `Walking`, `Standing`, `Fall`, `Lying_Down`.
- Uu tien demo real-time muot, co tracking ID, co nhan hanh dong.
- Uu tien phat hien nga tot hon viec classify hoan hao moi frame.
- Co file debug de chung minh loi den tu model, runtime, keypoints hay tracking.

## 4. Kien truc tong the

Luong xu ly dang chu:

1. Video hoac webcam dau vao.
2. YOLOv8 pose detect nguoi va keypoints.
3. Ultralytics tracking voi ByteTrack hoac BoT-SORT de gan `track_id`.
4. Moi `track_id` co rolling buffer keypoints.
5. Buffer du dai se tao sequence `(128, 69)`.
6. Action model du doan nhan hanh dong.
7. Runtime postprocess lam smoothing, fall gating, label hold va guardrail.
8. PyQt UI hien preview, bbox, skeleton, label, confidence va summary.
9. Output gom annotated video, run summary va fall debug timeline JSON.

Thanh phan chinh:

| Thanh phan | File | Vai tro |
|---|---|---|
| UI desktop | `pyqt_app.py` | Tao giao dien PyQt6, doc config UI, chay inference worker, hien preview va summary |
| Runtime shared | `src/runtime_shared.py` | Load pose/action model, prepare sequence, ActionRecognizerLite, rule runtime, debug state |
| Model action Torch | `train_professional_v3.py` | Dinh nghia Bi-GRU + Self-Attention va training pipeline |
| Feature/action helpers | `src/action_model_common.py` | Label map, ExtraTrees feature specs, color map |
| Module A | `src/module_a_detect.py` | Detect nguoi bang YOLOv8, khong tracking/action |
| Module B | `src/module_b_botsort_stable.py` | Tracking only voi BoT-SORT va memory bank |
| Module C | `src/module_c_action.py` | CLI legacy cho pose + tracking + action |
| Main CLI | `main.py` | Entry point chon mode `action` hoac `track` |
| Benchmark headless | `run_headless_profile_benchmark.py` | Chay cac profile Fast/Balanced/Quality khong can mo UI |

## 5. Cong nghe su dung

Nhom computer vision:

- `Ultralytics YOLOv8`: detect nguoi va pose estimation.
- `YOLOv8n-pose`: model pose nhe, hop voi laptop/RTX 3050.
- `OpenCV`: doc video, ghi video, ve bbox/skeleton, preview frame.
- `ByteTrack`, `BoT-SORT`: tracking ID.

Nhom machine learning:

- `PyTorch`: train va inference Bi-GRU action model.
- `scikit-learn`: ExtraTrees classifier va cac pipeline benchmark cu.
- `NumPy`, `SciPy`, `Pandas`: xu ly sequence, feature, metadata.

Nhom UI va runtime:

- `PyQt6`: desktop app.
- `TensorRT`: tuy chon tang toc pose inference neu dung `yolov8n-pose.engine`.
- `CUDA`: tang toc PyTorch/TensorRT tren GPU NVIDIA.

## 6. Cau truc thu muc quan trong

| Duong dan | Y nghia |
|---|---|
| `pyqt_app.py` | App UI chinh de test va demo |
| `src/` | Module runtime, detection, tracking, action |
| `config/` | Config tracker, detect va data train-ready |
| `config/data/` | Dataset da xu ly, dataset external, train-ready dataset |
| `VideoTest/` | Video test thu cong cho nga, ngoi, di bo |
| `runs/` | Ket qua train, output debug, annotated video, model artifact |
| `runs/qt_outputs/` | Fall debug timeline va video output tu UI |
| `runs/active_action_model_path.txt` | File chon action model active cho runtime |
| `TEST_GUIDELINES.md` | Huong dan chay app va benchmark |

## 7. Pipeline runtime trong PyQt6

File trung tam: `pyqt_app.py`.

Lop quan trong:

- `RuntimeConfig`: gom tat ca tham so runtime tu UI.
- `InferenceWorker`: QThread chay inference rieng de UI khong bi freeze.
- `_build_recognizer()`: tao `ActionRecognizerLite` tu action model path.
- `_run_inference()`: vong lap doc frame, detect/track, predict action, ve output va tao summary.
- `_apply_profile()`: set nhanh profile Fast/Balanced/Quality.

Thong so runtime quan trong:

- `det_conf`: confidence threshold cua YOLO pose.
- `imgsz`: kich thuoc input YOLO pose.
- `process_stride`: xu ly moi N frame.
- `preview_stride`: cap nhat preview moi N frame.
- `pred_stride`: so frame giua hai lan action prediction.
- `min_track_frames`: so frame toi thieu can gom cho track truoc khi predict.
- `smooth_window`: so prediction gan nhat de majority vote/smoothing.
- `fall_priority_prob`: nguong uu tien cho Fall.
- `fall_velocity_ratio`: nguong van toc roi theo chieu doc.
- `sitting_hold_frames`: so frame giu nhan Sitting.
- `max_det`: so nguoi toi da detect moi frame.

Luon can nho:

- UI co the hien FPS cao nhung action label van sai neu keypoints/sequence bi sai.
- Tang `imgsz` khong dam bao action dung hon, vi model action phu thuoc sequence va runtime rule.
- TensorRT tang toc pose, nhung khong tu dong sua loi classify hanh dong.

## 8. Runtime action recognizer

File trung tam: `src/runtime_shared.py`.

Class chinh: `ActionRecognizerLite`.

Nhiem vu:

- Load action model `.pth` hoac `.joblib`.
- Doc label map tu checkpoint/artifact.
- Chuan hoa label hien thi ve 5 nhan chinh.
- Duy tri buffer keypoints theo tung `track_id`.
- Tao input sequence `(128, 69)`.
- Chay model predict.
- Ap dung postprocess va physics/debug rule.
- Tra ve `(label_id, confidence, label_name)` cho UI.

Input sequence action model:

- Sequence dai 128 frame.
- Moi frame co 69 feature.
- 68 feature dau = 17 keypoints x 4 gia tri.
- 4 gia tri moi keypoint gom toa do da chuan hoa, velocity, acceleration.
- Feature cuoi = bbox aspect ratio.

Quality/debug state tinh trong runtime:

- `valid_ratio`: ti le keypoints hop le.
- `upper_body_ratio`: ti le keypoints than tren hop le.
- `lower_body_ratio`: ti le keypoints than duoi hop le.
- `downward_velocity`: toc do roi theo chieu doc cua body center.
- `center_motion_ratio`: muc di chuyen cua trong tam.
- `bbox_aspect_ratio`: ti le rong/cao cua bbox.
- `edge_contact`: co cham mep khung hinh hay khong.
- `fall_velocity`, `strong_fall_cue`, `moderate_fall_cue`, `lateral_fall_cue`: cac cue vat ly cho nga.
- `postprocess_reason`, `rescue_reason`: ly do runtime doi/giu nhan.

Nhan xet ky thuat:

- `ActionRecognizerLite` hien co nhieu rule hau xu ly. Cac rule nay giup bat nga va chan false positive, nhung neu qua manh co the lam nhan bi sai.
- Du an da tung co clean/simple mode de so sanh model raw voi runtime, nhung test thuc te cho thay clean 5-class co the kem hon legacy trong mot so video.
- Voi bai thuyet trinh, nen noi ro: he thong co ca model prediction va runtime safety logic, khong phai chi co neural network.

## 9. Action model active

File chon model active:

```text
runs/active_action_model_path.txt
```

Gia tri hien tai:

```text
runs/train_bigru_prod5_best_v1/final_safe_system.pth
```

Loai model:

- Backend: `torch`.
- Kien truc: Bi-GRU 3 layer + Multi-Head Self-Attention + MLP classifier.
- Input: `(batch, 128, 69)`.
- Output: logits cho 5 class.

Kien truc trong `train_professional_v3.py`:

- `LayerNorm` dau vao.
- Linear projection len hidden dim 128.
- GRU hai chieu, 3 layer.
- Self-attention pooling.
- MLP classifier.
- Training dung Focal Loss, class weight, AdamW, CosineAnnealingWarmRestarts, early stopping.

Label map cua model active:

| ID | Label |
|---:|---|
| 0 | Fall |
| 1 | Standing |
| 2 | Walking |
| 3 | Sitting |
| 4 | Lying_Down |

Metric da ghi trong artifact comparison:

| Dataset evaluation | Accuracy | Macro F1 |
|---|---:|---:|
| `train_ready_bigru_prod5_best_v1` validation | 0.96099 | 0.96095 |
| `train_ready_action_master_clean_v3_multicam_gmdcsa24` validation | 0.98579 | 0.98334 |
| `train_ready_action_repair_v6_five_action_round4_hardcases_gmdcsa24_full` validation | 0.98682 | 0.98269 |

So sanh voi ExtraTrees cu:

- ExtraTrees nhanh hon tren batch feature da tinh san.
- Bi-GRU cho accuracy/F1 cao hon tren cac validation set hien co.
- Runtime real video van co the sai neu pose/tracking/sequence khong giong du lieu validation.

## 10. Dataset train active

Dataset active cho Bi-GRU:

```text
config/data/train_ready_bigru_prod5_best_v1
```

Thong tin tu `dataset_card.json`:

- So mau: 5636.
- Shape: `(5636, 128, 69)`.
- So class: 5.
- Trang thai: `ready_pilot`.

Phan bo class:

| Class | So mau |
|---|---:|
| Fall | 1665 |
| Standing | 476 |
| Walking | 1720 |
| Sitting | 1012 |
| Lying_Down | 763 |

Nguon du lieu:

| Source | So mau |
|---|---:|
| UR_Fall | 2714 |
| Multicam | 1048 |
| Augment_Fall | 504 |
| Multicam_AllCams | 451 |
| GMDCSA24 | 403 |
| Unicomfacauca | 339 |
| NTU_pseudo | 177 |

Nhan xet ve dataset:

- `Walking` va `Fall` nhieu mau hon `Standing`.
- `Standing` chi co 476 mau, de gay thieu da dang khi test video thuc te.
- Du lieu train-ready la keypoint sequence, khong phai anh/video raw truc tiep.
- Neu pose extractor sinh keypoints khac tren video test, model se gap domain shift.
- `zero_frame_ratio_all` khoang 0.388, cho thay co nhieu frame/keypoint missing hoac padding. Dieu nay co the anh huong den class nhu Sitting/Standing khi bi che khuat.

## 11. Dataset va script xu ly du lieu

Nguon raw/processed dang co:

- `config/data/UR_Fall`: UR Fall dataset, nhieu canh ADL va fall.
- `config/data/Multicam`: Multicam fall/action dataset.
- `config/data/external/gmdcsa24`: du lieu ngoai them.
- `config/data/external/unicomfacauca`: du lieu ngoai them cho standing/walking/sitting/lying.
- `config/data/ntu_10_actions_filtered`: NTU subset/pseudo mapping.
- `VideoTest/`: video test thu cong, gom `Human Fall Detection Sample.mp4`, `ngoi.mp4`, video walking.

Script lien quan:

| Script | Vai tro |
|---|---|
| `extract_pose.py` | Trich pose/keypoints tu video/dataset |
| `data_prepare_v3.py` | Chuan bi sequence 128x69 cho training cu |
| `prepare_master_clean_dataset.py` | Tao clean dataset va tinh quality metrics |
| `prepare_action_hardcases_round2.py` | Tao hardcase augmentation cho walking occlusion/bending boundary |
| `repair_action_dataset.py` | Sua/tai cau truc dataset action |
| `integrate_multicam_allcams_master_clean.py` | Tich hop Multicam all-cams |
| `integrate_gmdcsa24_fall_transitions.py` | Tich hop fall transition tu GMDCSA24 |
| `integrate_unicomfacauca_action_repair.py` | Tich hop Unicomfacauca |
| `integrate_ntu_mapped.py` | Map NTU vao action dataset |
| `train_professional_v3.py` | Train Bi-GRU action model |
| `train_extratrees_action.py` | Train ExtraTrees action classifier |
| `mine_timeline_hardcases.py` | Mine loi tu fall debug timeline |
| `train_timeline_hardcase_calibrator.py` | Train calibrator tu hardcase timeline |

## 12. Cac profile runtime trong UI

Profile Fast:

- Tracker: ByteTrack custom.
- `imgsz`: 480.
- `process_stride`: 2.
- `preview_stride`: 3 trong UI hien tai.
- `pred_stride`: 1.
- `min_track_frames`: 12 trong UI hien tai.
- Muc tieu: muot, nhanh, phu hop demo real-time.

Profile RTX 3050 Balanced:

- Tracker: BoT-SORT custom.
- `imgsz`: 640.
- `process_stride`: 2.
- `preview_stride`: 3.
- `pred_stride`: 2.
- `min_track_frames`: 5.
- Muc tieu: can bang pose quality va toc do.

Profile Quality:

- Tracker: BoT-SORT custom.
- `imgsz`: 640 trong UI hien tai.
- `process_stride`: 1.
- `preview_stride`: 2.
- Co save output video mac dinh.
- Muc tieu: nhin frame day hon, nhung co the nang hon va khong chac chinh xac hon neu action model/runtime bi tre.

Nhan xet thuc nghiem tu cac lan test:

- Fast thuong cho cam giac muot hon va trong nhieu video co ve on hon Quality.
- Quality khong dam bao label dung hon, vi neu action queue/prediction cadence va sequence khong hop thi van sai.
- TensorRT giup FPS pose tang manh, nhung loi `Sitting`/`Walking`/`Fall` van la bai toan action model + runtime state.

## 13. Debug output va cach doc ket qua

Moi run PyQt video co the tao:

- Annotated video trong `runs/qt_outputs/`.
- Fall debug timeline JSON: `runs/qt_outputs/fall_debug_timeline_*.json`.
- Run summary hien trong UI.

Run summary quan trong:

- `Average FPS`: FPS trung binh toan run.
- `Live FPS EMA`: FPS hien tai da lam muot.
- `Unique Track IDs`: so ID duy nhat, neu qua cao co the tracking bi cat/vang.
- `Total Detections`: tong detection.
- `Frames With Detections`: so frame co detection.
- `Effective Pose ImgSz`: imgsz thuc su sau khi resolve TensorRT/PyTorch.
- `Effective Process Stride`: xu ly moi N frame.
- `Effective Action Pred Stride`: action predict moi N frame.
- `Action Requests/Completed`: so request action gui/hoan thanh.
- `Action Stale Results Dropped`: ket qua action bi bo vi qua cu.
- `Action Queue Busy Frames`: so frame action predictor dang ban.
- `Pose Backend`: PyTorch hay TensorRT.
- `Action Backend`: torch hay extratrees.
- `Action Counts`: tong so label da hien thi.

Timeline record quan trong:

- `frame`, `sec`, `tid`: vi tri thoi gian va ID.
- `label`, `conf`: nhan hien thi va confidence.
- `fall_cue`, `fall_vel`: tin hieu runtime ve nga.
- `down_vel`: van toc roi theo chieu doc.
- `bbox_ar`: bbox aspect ratio.
- `resc`: ly do rescue/postprocess neu co.

Cach dung debug:

- Neu raw/model doan dung nhung final sai: loi do runtime postprocess.
- Neu raw/model da sai lien tuc: loi do model, data, keypoints hoac domain shift.
- Neu label nhay khi ID doi: loi tracking/scene cut/track buffer.
- Neu chi sai luc ra mep khung hinh: loi edge/occlusion guard.

## 14. Tai sao train/test tren data van co the sai

Day la diem quan trong nen noi trong thuyet trinh.

Ly do 1: Data train la pose sequence, khong phai video raw.

- Neu YOLO pose tren video test tao keypoints khac voi keypoints luc build dataset, model se thay input khac.
- Che khuat chan, mat hong, mat dau goi lam sequence bien dang.

Ly do 2: Validation co the bi optimistic.

- Stratified split co the chia cac segment gan nhau vao train/val.
- Grouped validation tot hon, nhung van khong bao phu moi goc quay thuc te.

Ly do 3: Montage/cut canh lam nhan cu bi keo sang canh moi.

- Tracking ID co the bi reset, buffer chua du frame, label cu duoc giu.
- Scene cut nhanh lam van toc/center motion bi nham la fall cue.

Ly do 4: Class `Standing`, `Walking`, `Sitting` rat gan nhau ve skeleton.

- Neu chi thay than tren, nguoi dung im hoac di cham de bi nham.
- Nguoi ngoi thang lung co bbox gan voi standing neu chan bi che.

Ly do 5: Runtime rule co hai mat.

- Rule giup bat nga nhanh va chan false fall.
- Rule qua manh co the ep `Walking` thanh `Standing`, ep `Sitting` thanh `Fall`, hoac giu nhan cu qua lau.

## 15. Van de hien tai da quan sat

Tu lich su test gan day, cac loi noi bat:

- Video nga: mot so pha nga van bi gan `Walking`, `Sitting` hoac `Lying_Down`.
- Sau khi nga: co luc `Fall` bi chuyen sang `Sitting` qua som.
- Video ngoi: luc nguoi ra khoi man hinh co false `Fall`.
- Video di bo: nguoi dung yen o goc khung hinh co the bi gan `Walking`.
- `Standing` va `Walking` con nham nhau khi nguoi di cham, dung quay lung, hoac keypoints chan thieu.
- `Sitting` co the bi mat khi chan bi che hoac bbox khong ro dang ngoi.

Nguyen nhan kha nang cao:

- Khong chi do model.
- Cung khong chi do runtime.
- Loi den tu tong hop: pose missing, tracking/buffer, data distribution, cut canh va postprocess.

## 16. Diem manh cua du an

- Co pipeline end-to-end tu video den annotated output.
- Co UI PyQt6 de demo truc tiep.
- Ho tro CUDA/TensorRT cho pose inference.
- Co action model deep learning rieng, khong chi rule-based.
- Co debug timeline chi tiet de phan tich loi.
- Co nhieu script data repair, hardcase mining va retraining.
- Co benchmark headless de so sanh profile lap lai.
- Co thiet ke modular: Module A detect, Module B tracking, Module C action.

## 17. Gioi han hien tai

- Accuracy runtime chua on dinh tren moi video thuc te.
- Runtime postprocess con phuc tap va co nhieu rule lich su.
- Dataset con lech class, dac biet `Standing` it hon `Walking` va `Fall`.
- Mot so external data co the khong trung domain voi video test.
- Demo tren video ghep/cut nhanh de tao loi state carry-over.
- Chua co benchmark dinh luong co ground truth theo frame cho tung video test UI.

## 18. Huong cai tien ky thuat

Huong uu tien 1: Tach danh gia model raw va runtime final.

- Luu ca `raw_label_name` va `final_label_name`.
- Tao confusion matrix rieng cho raw model.
- Tao confusion matrix rieng cho final runtime.
- Neu raw dung, sua runtime.
- Neu raw sai, retrain/data repair.

Huong uu tien 2: Tao benchmark co ground truth cho video test.

- Gan nhan theo doan: start frame, end frame, action.
- Chay app/headless va so sanh theo frame/segment.
- Dung metric: per-class precision, recall, F1; fall miss rate; false fall rate.

Huong uu tien 3: Bo sung du lieu dung voi loi that.

- Walking cham, walking bi che mot phan.
- Standing quay lung, dung o mep khung hinh.
- Sitting thang lung, sitting bi che chan, sitting ra khoi khung.
- Fall tu ghe, fall ngang, fall nhe, fall ve truoc, fall ve sau.
- Lying down khong phai fall.

Huong uu tien 4: Don gian hoa runtime rule.

- Giu rule bat buoc cho safety: fall hold, edge false fall block, ID reset.
- Giam cac rule ep label qua manh cho walking/standing/sitting.
- Moi thay doi rule phai co benchmark truoc/sau.

Huong uu tien 5: Cai tien tracking/scene cut.

- Scene cut reset buffer ro rang hon.
- Khi track moi xuat hien sau cut canh, khong keo label cu.
- Khi nguoi ra mep khung hinh, giam confidence thay vi ep `Fall`.

## 19. Ke hoach demo de thuyet trinh

Demo nen gom 4 buoc:

1. Mo app PyQt6.
2. Chay video nga `VideoTest/Human Fall Detection Sample.mp4`.
3. Cho xem summary: FPS, backend, action counts, fall timeline.
4. Mo mot timeline JSON va giai thich cac cot `fall_cue`, `down_vel`, `bbox_ar`, `rescue_reason`.

Nen chuan bi them:

- Mot video ngoi: `VideoTest/ngoi.mp4`.
- Mot video walking: video walking trong `VideoTest/`.
- Mot slide ve loi hien tai de hoi dap minh bach.

Len noi ro trong demo:

- Day la prototype nghien cuu/ky thuat, khong phai san pham y te/an ninh san sang production.
- Ket qua phu thuoc goc quay, anh sang, occlusion va chat luong pose.

## 20. De cuong slide de xay presentation

Slide 1: Tieu de

- Multi-Person Tracking and Pose-based Action Analysis in Video.
- Muc tieu: tracking nguoi va nhan dien hanh dong dua tren pose.

Slide 2: Bai toan

- Camera/video dau vao.
- Nhieu nguoi, nhieu ID.
- Can nhan dien 5 hanh dong.
- Fall detection la case quan trong nhat.

Slide 3: Thach thuc

- Occlusion.
- Goc quay khac nhau.
- Cut canh nhanh.
- Standing/Walking/Sitting giong nhau.
- Fall co nhieu kieu.

Slide 4: Tong quan pipeline

- Video input.
- YOLOv8 pose.
- ByteTrack/BoT-SORT.
- Keypoint buffer.
- Bi-GRU action model.
- Runtime postprocess.
- PyQt UI/output.

Slide 5: Detection va Tracking

- YOLOv8 pose detect nguoi va 17 keypoints.
- ByteTrack/BoT-SORT gan ID.
- ID giup gom sequence theo tung nguoi.

Slide 6: Action Recognition Model

- Input `(128, 69)`.
- Bi-GRU 3 layer.
- Self-attention pooling.
- 5 output classes.
- Active model path.

Slide 7: Dataset

- 5636 samples.
- 5 classes.
- Nguon: UR_Fall, Multicam, GMDCSA24, Unicomfacauca, NTU_pseudo.
- Class distribution.

Slide 8: Runtime va UI

- PyQt6 app.
- Fast/Balanced/Quality profile.
- TensorRT/PyTorch pose backend.
- Run summary va debug timeline.

Slide 9: Ket qua training/evaluation

- BigRU val accuracy/F1 tren cac validation set.
- So sanh voi ExtraTrees.
- Luu y validation khong dong nghia voi video thuc te dung 100%.

Slide 10: Loi thuc te va phan tich

- Fall co luc thanh Sitting/Walking.
- Sitting ra mep khung co false Fall.
- Standing/Walking nham nhau.
- Nguyen nhan: pose missing, data domain shift, runtime state.

Slide 11: Huong cai tien

- Ground-truth video benchmark.
- Mining hardcases tu timeline.
- Bo sung data dung loi that.
- Don gian hoa runtime rule.
- Cai tien scene cut/edge handling.

Slide 12: Ket luan

- He thong da co pipeline end-to-end va UI demo.
- Thanh cong lon: detect, track, sequence action, debug.
- Viec con lai: chuan hoa benchmark va cai tien model/runtime bang hardcase co nhan dung.

## 21. Kich ban thuyet trinh ngan

Mo dau:

"Du an cua em tap trung vao bai toan nhan dien hanh dong nguoi trong video. Thay vi classify truc tiep tren anh RGB, he thong trich xuat pose cua tung nguoi, tracking ID theo thoi gian va dung chuoi keypoints de du doan hanh dong."

Giai thich pipeline:

"Moi frame duoc dua qua YOLOv8 pose de lay bbox va 17 keypoints. Sau do tracker gan ID cho tung nguoi. Voi moi ID, he thong gom keypoints thanh buffer 128 frame, bien doi thanh vector 69 feature moi frame va dua vao model Bi-GRU Self-Attention."

Giai thich vi sao can tracking:

"Neu khong co tracking ID, ta khong biet keypoints o frame sau thuoc ve nguoi nao. Tracking ID giup model nhin duoc hanh dong lien tuc cua tung nguoi, thay vi chi nhin tung frame rieng le."

Giai thich model:

"Model action hien tai la Bi-GRU hai chieu 3 layer ket hop self-attention. GRU hoc thay doi theo thoi gian, attention giup model tap trung vao cac frame quan trong trong sequence, vi du khoanh khac bat dau nga."

Giai thich runtime:

"Runtime khong chi lay output model roi hien thi ngay. He thong co smoothing va mot so guardrail de giam nhap nhay, giu nhan Fall trong vai frame va chan false Fall khi nguoi ra khoi khung hinh."

Giai thich ket qua:

"Tren validation set, model Bi-GRU dat khoang 96% accuracy va macro F1 tren dataset production 5-label. Tuy nhien khi chay video thuc te, loi van xay ra do pose missing, cut canh, tracking ID va domain shift."

Ket luan:

"Diem manh cua du an la da co pipeline end-to-end va debug timeline de truy vet loi. Huong tiep theo la tao benchmark video co ground truth theo frame, mine hardcases tu timeline va retrain bang cac case sai that."

## 22. Cau hoi co the bi hoi va cach tra loi

Hoi: Tai sao khong dung CNN/Video Transformer truc tiep tren RGB?

Tra loi: Cach pose-based nhe hon, de chay real-time hon va giai thich duoc bang keypoints. Tuy nhien no phu thuoc vao chat luong pose, nen khi che khuat keypoints thi accuracy giam.

Hoi: Tai sao validation cao nhung demo van sai?

Tra loi: Validation duoc tinh tren pose sequence da xu ly. Video demo co detection, tracking, keypoint missing, cut canh va runtime postprocess. Loi runtime la loi ca pipeline, khong chi loi model.

Hoi: TensorRT co lam action dung hon khong?

Tra loi: TensorRT chu yeu tang toc pose inference. No giup FPS cao hon, nhung action accuracy van phu thuoc action model, keypoint sequence va postprocess.

Hoi: Vi sao `Standing` va `Walking` hay nham?

Tra loi: Hai hanh dong nay rat gan nhau neu nguoi di cham, dung gan nhu yen, hoac keypoints chan bi che. Can them feature van toc theo ID, du lieu walking cham va standing nhieu goc quay hon.

Hoi: Vi sao `Sitting` co luc thanh `Fall`?

Tra loi: Khi nguoi ra mep khung hinh hoac bi mat chan, trong tam/bbox co the thay doi dot ngot. Runtime co the hieu do la fall cue. Can edge-exit guard va du lieu sitting exit-frame.

Hoi: He thong co san sang production khong?

Tra loi: Chua. Hien tai la prototype/de tai nghien cuu co UI demo va debug. Can benchmark co ground truth, test tren nhieu camera va giam false positive/false negative truoc khi production.

## 23. Lenh chay quan trong

Chay app PyQt6 bang env TensorRT:

```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .trt-export-venv/bin/activate
python pyqt_app.py
```

Chay fallback khong TensorRT:

```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .venv/bin/activate
python pyqt_app.py
```

Chay benchmark headless:

```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .trt-export-venv/bin/activate
python run_headless_profile_benchmark.py \
  --video "VideoTest/Human Fall Detection Sample.mp4" \
  --profiles fast,balanced,quality
```

Kiem tra action model active:

```bash
cat runs/active_action_model_path.txt
```

Train lai Bi-GRU voi dataset active:

```bash
python train_professional_v3.py \
  --data_dir config/data/train_ready_bigru_prod5_best_v1 \
  --save_dir runs/train_bigru_prod5_best_v1
```

## 24. Ket luan cuoi

Du an nay co gia tri o cho no khong chi la mot classifier don gian. No la mot he thong computer vision day du gom detection, pose estimation, tracking, temporal action recognition, UI real-time, debug timeline va training pipeline.

Diem can nhan manh khi thuyet trinh:

- Pipeline da chay end-to-end.
- Co the demo tren video/webcam.
- Co GPU/TensorRT acceleration.
- Co model Bi-GRU 5-class active.
- Co du lieu train tu nhieu nguon.
- Co co che debug de biet loi den tu model hay runtime.

Diem can noi thang:

- Runtime action recognition van chua on dinh tren moi case thuc te.
- `Sitting`, `Standing`, `Walking` va cac pha nga dac biet van la hard cases.
- Can benchmark co ground truth theo video truoc khi khang dinh do chinh xac production.

Thong diep chinh cho slide:

"He thong da xay duoc nen tang nhan dien hanh dong dua tren pose va tracking ID. Ket qua tot tren validation, co UI demo va debug pipeline. Buoc tiep theo la chuan hoa benchmark video thuc te va retrain bang hardcases de nang do on dinh runtime."
