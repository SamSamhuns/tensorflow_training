# in-built models are available at: https://www.tensorflow.org/api_docs/python/tf/keras/applications
from tf_train.model.models import facenet
import tensorflow as tf

# Note the types of the models must be derived from the instantiated
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
             'weights': "model_store/facenet_weights/facenet_keras.h5",
         },
         'gives_logits': False,
         'parent_module': facenet,
         'module': facenet.load_facenet_model,
         'final_module': facenet.FacenetPred},
    'Densenet121':
        {'type': 'densenet121',
         'args': {
             'include_top': True,
             'input_shape': (224, 224, 3),
             'weights': None,
             'pooling': None
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.densenet,
         'module': tf.keras.applications.DenseNet121,
         'final_module': None},
    'EfficientNetB4':
        {'type': 'efficientnetb4',
         'args': {
             'include_top': True,
             'input_shape': (528, 528, 3),
             'weights': None,
             'pooling': None
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet,
         'module': tf.keras.applications.EfficientNetB4,
         'final_module': None},
    'EfficientNetB6':
        {'type': 'efficientnetb6',
         'args': {
             'include_top': True,
             'input_shape': (528, 528, 3),
             'weights': None,
             'pooling': None
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet,
         'module': tf.keras.applications.EfficientNetB6,
         'final_module': None},
    'InceptionResNetV2':
        {'type': 'inception_resnet_v2',
         'args': {
             'include_top': True,
             'input_shape': (299, 299, 3),
             'weights': None,
             'pooling': None
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.inception_resnet_v2,
         'module': tf.keras.applications.InceptionResNetV2,
         'final_module': None},
    'InceptionV3':
        {'type': 'inceptionV3',
         'args': {
             'include_top': True,
             'input_shape': (299, 299, 3),
             'weights': None,
             'pooling': None
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.inception_v3,
         'module': tf.keras.applications.InceptionV3,
         'final_module': None},
    'MobileNetV2':
        {'type': 'mobilenetv2_1.00_224',
         'args': {
             'include_top': True,
             'input_shape': (224, 224, 3),
             'weights': None,
             'pooling': None
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.mobilenet_v2,
         'module': tf.keras.applications.MobileNetV2,
         'final_module': None},
    'MobileNetV3Large':
        {'type': 'MobilenetV3large',
         'args': {
             'include_top': True,
             'input_shape': (224, 224, 3),
             'weights': None,
             'pooling': None
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.mobilenet_v3,
         'module': tf.keras.applications.MobileNetV3Large,
         'final_module': None},
    'Xception':
        {'type': 'xception',
         'args': {
             'include_top': True,
             'input_shape': (299, 299, 3),
             'weights': None,
             'pooling': None
         },
         'gives_logits': False,
         'parent_module': tf.keras.applications.xception,
         'module': tf.keras.applications.Xception,
         'final_module': None}
}
