# in-built models are available at: https://www.tensorflow.org/api_docs/python/tf/keras/applications
from custom_models import facenet
import tensorflow as tf


model_info_dict = {
    'facenet':
        {'name': 'facenet',
         'in_w': 160,
         'in_h': 160,
         'in_c': 3,
         'gives_logits': False,
         'parent_module': facenet,
         'module': facenet.get_facenet_model},
    'densenet121':
        {'name': 'densenet121',
         'in_w': 224,
         'in_h': 224,
         'in_c': 3,
         'gives_logits': False,
         'parent_module': tf.keras.applications.densenet,
         'module': tf.keras.applications.DenseNet121},
    'EfficientNetB4':
        {'name': 'efficientnetb4',
         'in_w': 528,
         'in_h': 528,
         'in_c': 3,
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet,
         'module': tf.keras.applications.EfficientNetB4},
    'EfficientNetB6':
        {'name': 'efficientnetb6',
         'in_w': 528,
         'in_h': 528,
         'in_c': 3,
         'gives_logits': False,
         'parent_module': tf.keras.applications.efficientnet,
         'module': tf.keras.applications.EfficientNetB6},
    'InceptionResNetV2':
        {'name': 'inception_resnet_v2',
         'in_w': 299,
         'in_h': 299,
         'in_c': 3,
         'gives_logits': False,
         'parent_module': tf.keras.applications.inception_resnet_v2,
         'module': tf.keras.applications.InceptionResNetV2},
    'InceptionV3':
        {'name': 'inceptionV3',
         'in_w': 299,
         'in_h': 299,
         'in_c': 3,
         'gives_logits': False,
         'parent_module': tf.keras.applications.inception_v3,
         'module': tf.keras.applications.InceptionV3},
    'MobileNetV2':
        {'name': 'mobilenet_v2',
         'in_w': 224,
         'in_h': 224,
         'in_c': 3,
         'gives_logits': False,
         'parent_module': tf.keras.applications.mobilenet_v2,
         'module': tf.keras.applications.MobileNetV2},
    'MobileNetV3Large':
        {'name': 'MobilenetV3large',
         'in_w': 224,
         'in_h': 224,
         'in_c': 3,
         'gives_logits': False,
         'parent_module': tf.keras.applications.mobilenet_v3,
         'module': tf.keras.applications.MobileNetV3Large},
    'Xception':
        {'name': 'xception',
         'in_w': 299,
         'in_h': 299,
         'in_c': 3,
         'gives_logits': False,
         'parent_module': tf.keras.applications.xception,
         'module': tf.keras.applications.Xception}
}
