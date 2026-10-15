import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
_cache = {}


def _parts(model):
    """Split the saved model into: image -> last conv features, and the final Dense layer."""
    key = id(model)
    if key not in _cache:
        base = next(l for l in model.layers if "mobilenet" in l.name.lower())
        conv_model = tf.keras.Model(base.input, base.get_layer("out_relu").output)
        _cache[key] = (conv_model, model.layers[-1])
    return _cache[key]


def make_heatmap(img_uint8, model, class_idx=None):
    """img_uint8: (224, 224, 3) RGB array. Returns (heatmap in [0,1] of size 224x224, class index)."""
    conv_model, dense = _parts(model)
    x = tf.convert_to_tensor(img_uint8[None, ...].astype("float32"))
    x = x / 127.5 - 1.0  # same scaling as inside the trained model

    with tf.GradientTape() as tape:
        conv_out = conv_model(x, training=False)      # (1, 7, 7, 1280)
        tape.watch(conv_out)
        preds = dense(tf.reduce_mean(conv_out, axis=[1, 2]))
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]

    grads = tape.gradient(score, conv_out)             # how much each feature map matters
    weights = tf.reduce_mean(grads, axis=(1, 2))       # importance weight per feature map
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()                      # keep only positive influence
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., None], IMG_SIZE).numpy()[..., 0]
    return np.clip(cam, 0, 1), class_idx


def _jet(x):
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay(img_uint8, cam, strength=0.65):
    """Blend a jet-colored heatmap onto the image; low-attention areas stay clear."""
    heat = _jet(cam) * 255
    a = (strength * cam)[..., None]
    out = img_uint8.astype("float32") * (1 - a) + heat * a
    return np.clip(out, 0, 255).astype("uint8")
