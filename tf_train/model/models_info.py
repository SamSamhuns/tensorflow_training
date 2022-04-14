# in-built models are available at: https://www.tensorflow.org/api_docs/python/tf/keras/applications
from tf_train.model.models import facenet, resnet50_places365
from tf_train.model.models import base
import tensorflow as tf

# Note: the model msut match the name of the final_module if it is not None
# if final_module is None, type must be derived from the instantiated
# model eg. tf.keras.applications.DenseNet121().name => densenet121
# modules and parent modules should also not be initialized in this file
# use module=foo instead if module=foo()
# weights=None,  # None, imagenet
# pooling=None,  # None, avg, max
model_info_dict = {
    'Facenet':
        {'type': 'facenet_pred',
         'args': {
             'input_shape': (160, 160, 3),
             'weights': "model_store/facenet_weights/facenet_keras_p38"
         },
         'gives_logits': False,
         'parent_module': facenet,
         'module': facenet.load_facenet_model,
         'final_module': facenet.FacenetPred},
    'Resnet50_places365':
        {'type': 'resnet50_places365_pred',
         'args': {
             'input_shape': (224, 224, 3),
             'weights': "model_store/resnet50_places365_weights/resnet50_places365"
         },
         'gives_logits': False,
         'parent_module': resnet50_places365,
         'module': resnet50_places365.load_resnet50_places365_model,
         'final_module': resnet50_places365.Resnet50Places365Pred},
    'Densenet121':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (224, 224, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.densenet,
         'module': tf.keras.applications.DenseNet121,
         'final_module': base.Classifier},
    'EfficientNetB0':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (224, 224, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet,
         'module': tf.keras.applications.EfficientNetB0,
         'final_module': base.Classifier},
    'EfficientNetB2':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (260, 260, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet,
         'module': tf.keras.applications.EfficientNetB2,
         'final_module': base.Classifier},
    'EfficientNetB4':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (528, 528, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet,
         'module': tf.keras.applications.EfficientNetB4,
         'final_module': base.Classifier},
    'EfficientNetB6':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (528, 528, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet,
         'module': tf.keras.applications.EfficientNetB6,
         'final_module': base.Classifier},
    'EfficientNetV2B0':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (224, 224, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet_v2,
         'module': tf.keras.applications.EfficientNetV2B0,
         'final_module': base.Classifier},
    'EfficientNetV2B1':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (240, 240, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet_v2,
         'module': tf.keras.applications.EfficientNetV2B1,
         'final_module': base.Classifier},
    'EfficientNetV2B2':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (260, 260, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet_v2,
         'module': tf.keras.applications.EfficientNetV2B2,
         'final_module': base.Classifier},
    'EfficientNetV2B3':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (300, 300, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet_v2,
         'module': tf.keras.applications.EfficientNetV2B3,
         'final_module': base.Classifier},
    'EfficientNetV2S':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (384, 384, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet_v2,
         'module': tf.keras.applications.EfficientNetV2S,
         'final_module': base.Classifier},
    'EfficientNetV2M':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (480, 480, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet_v2,
         'module': tf.keras.applications.EfficientNetV2M,
         'final_module': base.Classifier},
    'EfficientNetV2L':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (480, 480, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet_v2,
         'module': tf.keras.applications.EfficientNetV2L,
         'final_module': base.Classifier},
    'InceptionResNetV2':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (299, 299, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.inception_resnet_v2,
         'module': tf.keras.applications.InceptionResNetV2,
         'final_module': base.Classifier},
    'InceptionV3':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (299, 299, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.inception_v3,
         'module': tf.keras.applications.InceptionV3,
         'final_module': base.Classifier},
    'MobileNetV2':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (224, 224, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.mobilenet_v2,
         'module': tf.keras.applications.MobileNetV2,
         'final_module': base.Classifier},
    'MobileNetV3Large':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (224, 224, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.mobilenet_v3,
         'module': tf.keras.applications.MobileNetV3Large,
         'final_module': base.Classifier},
    'MobileNetV3Small':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (224, 224, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.mobilenet_v3,
         'module': tf.keras.applications.MobileNetV3Small,
         'final_module': base.Classifier},
    'Xception':
        {'type': 'classifier',
         'args': {
             'include_top': False,
             'input_shape': (299, 299, 3),
             'weights': 'imagenet',
             'pooling': 'avg'
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.xception,
         'module': tf.keras.applications.Xception,
         'final_module': base.Classifier}
}
