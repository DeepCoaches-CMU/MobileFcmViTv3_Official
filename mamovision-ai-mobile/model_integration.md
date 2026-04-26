# Model Integration

The app is fully wired for real inference. All that remains is exporting the checkpoint and rebuilding.

---

## Step 1 — Export checkpoint to ONNX

Run from the **project root** (one level above this folder):

```bash
python export_onnx.py \
  --checkpoint checkpoints/<your_run>/model_best.pth.tar \
  --output mamovision-ai-mobile/assets/models/mobilefcmvitv3.onnx
```

This exports FP32 and then automatically quantizes to INT8 (~5–6 MB). Both files are saved; the INT8 version is what gets named `mobilefcmvitv3.onnx` in assets.

To skip quantization:
```bash
python export_onnx.py --checkpoint ... --output ... --no-quantize
```

---

## Step 2 — Rebuild the app

A full native rebuild is required because `onnxruntime-react-native` is a native module (Expo Go will not work):

```bash
cd mamovision-ai-mobile
npx expo run:android
```

---

## That's it

Once the app starts, the ONNX session loads in the background. Model status in **Settings → Developer Options** will change from `loading` → `ready` when it's done.

**Class order** in the ONNX output: `[benign=0, malignant=1, normal=2]`  
The app reorders to `[normal, benign, malignant]` internally before BI-RADS mapping.
