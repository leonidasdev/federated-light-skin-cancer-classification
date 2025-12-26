"""
Definición de la arquitectura CNN ligera para clasificación de lesiones cutáneas.
Inspirada en Mamun et al. (2025).

Arquitectura:
- 3 bloques convolucionales (32, 64, 128 filtros)
- MaxPooling después de cada bloque
- Flatten
- Dense(256, ReLU, Dropout)
- Softmax(7 clases)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

from config.config import MODEL_CONFIG


def create_cnn_model(input_shape=None, num_classes=None, dropout_rate=None):
    """
    Crea el modelo CNN ligero para clasificación de lesiones cutáneas.
    
    Args:
        input_shape (tuple): Forma de entrada (height, width, channels)
        num_classes (int): Número de clases de salida
        dropout_rate (float): Tasa de dropout
    
    Returns:
        keras.Model: Modelo compilado
    """
    # Usar valores de configuración si no se especifican
    if input_shape is None:
        input_shape = MODEL_CONFIG['input_shape']
    if num_classes is None:
        num_classes = MODEL_CONFIG['num_classes']
    if dropout_rate is None:
        dropout_rate = MODEL_CONFIG['dropout_rate']
    
    # Entrada
    inputs = layers.Input(shape=input_shape, name='input_images')
    x = inputs
    
    # Bloque Convolucional 1: 32 filtros
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=l2(0.001),
        name='conv1_1'
    )(x)
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=l2(0.001),
        name='conv1_2'
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    
    # Bloque Convolucional 2: 64 filtros
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=l2(0.001),
        name='conv2_1'
    )(x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=l2(0.001),
        name='conv2_2'
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    
    # Bloque Convolucional 3: 128 filtros
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=l2(0.001),
        name='conv3_1'
    )(x)
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=l2(0.001),
        name='conv3_2'
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    
    # Almacenar para Grad-CAM (última capa convolucional)
    last_conv_layer = x
    
    # Aplanar
    x = layers.Flatten(name='flatten')(x)
    
    # Capa densa con dropout
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=l2(0.001),
        name='dense1'
    )(x)
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    
    # Capa de salida (softmax)
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(x)
    
    # Crear modelo
    model = models.Model(inputs=inputs, outputs=outputs, name='LightCNN_SkinCancer')
    
    return model


def compile_model(model, learning_rate=None, loss_function=None):
    """
    Compila el modelo con optimizador, pérdida y métricas.
    
    Args:
        model (keras.Model): Modelo a compilar
        learning_rate (float): Tasa de aprendizaje
        loss_function (str): Función de pérdida
    
    Returns:
        keras.Model: Modelo compilado
    """
    from config.config import TRAINING_CONFIG
    
    # Usar valores de configuración si no se especifican
    if learning_rate is None:
        learning_rate = TRAINING_CONFIG['learning_rate']
    if loss_function is None:
        loss_function = TRAINING_CONFIG['loss_function']
    
    # Configurar optimizador
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=TRAINING_CONFIG['optimizer_params']['beta_1'],
        beta_2=TRAINING_CONFIG['optimizer_params']['beta_2'],
        epsilon=TRAINING_CONFIG['optimizer_params']['epsilon']
    )
    
    # Configurar función de pérdida
    if loss_function == 'focal_loss':
        # TODO: Implementar focal loss para datos desbalanceados
        loss = keras.losses.CategoricalCrossentropy()
    else:
        loss = keras.losses.CategoricalCrossentropy()
    
    # Métricas
    metrics = [
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc', multi_label=True)
    ]
    
    # Compilar
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_model_summary(model):
    """
    Obtiene un resumen del modelo.
    
    Args:
        model (keras.Model): Modelo
    
    Returns:
        str: Resumen del modelo
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def count_parameters(model):
    """
    Cuenta los parámetros totales y entrenables del modelo.
    
    Args:
        model (keras.Model): Modelo
    
    Returns:
        dict: Diccionario con conteo de parámetros
    """
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def create_focal_loss(alpha=0.25, gamma=2.0):
    """
    Crea función de focal loss para manejar desbalance de clases.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (float): Factor de balance
        gamma (float): Factor de enfoque
    
    Returns:
        function: Función de focal loss
    """
    def focal_loss(y_true, y_pred):
        """
        Calcula focal loss.
        
        Args:
            y_true: Etiquetas verdaderas (one-hot)
            y_pred: Predicciones del modelo
        
        Returns:
            tensor: Loss calculado
        """
        # TODO: Implementar focal loss
        # epsilon para estabilidad numérica
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calcular cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calcular focal weight
        weight = alpha * tf.pow((1 - y_pred), gamma)
        
        # Focal loss
        loss = weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    return focal_loss


def get_last_conv_layer_name(model):
    """
    Obtiene el nombre de la última capa convolucional (para Grad-CAM).
    
    Args:
        model (keras.Model): Modelo
    
    Returns:
        str: Nombre de la última capa convolucional
    """
    conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
    if conv_layers:
        return conv_layers[-1]
    return None


# ==================== FUNCIONES DE UTILIDAD ====================

def save_model_architecture(model, filepath='model_architecture.png'):
    """
    Guarda la arquitectura del modelo como imagen.
    
    Args:
        model (keras.Model): Modelo
        filepath (str): Ruta donde guardar la imagen
    """
    # TODO: Implementar visualización de arquitectura
    try:
        keras.utils.plot_model(
            model,
            to_file=filepath,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96
        )
        print(f"Arquitectura guardada en: {filepath}")
    except Exception as e:
        print(f"Error al guardar arquitectura: {e}")


def print_model_info(model):
    """
    Imprime información detallada del modelo.
    
    Args:
        model (keras.Model): Modelo
    """
    print("\n" + "=" * 60)
    print("INFORMACIÓN DEL MODELO")
    print("=" * 60)
    
    # Resumen
    model.summary()
    
    # Parámetros
    params = count_parameters(model)
    print("\nPARÁMETROS:")
    print(f"  Total: {params['total']:,}")
    print(f"  Entrenables: {params['trainable']:,}")
    print(f"  No entrenables: {params['non_trainable']:,}")
    
    # Última capa convolucional
    last_conv = get_last_conv_layer_name(model)
    print(f"\nÚltima capa convolucional (Grad-CAM): {last_conv}")
    
    print("=" * 60 + "\n")


# ==================== TESTING ====================

if __name__ == '__main__':
    # Crear y mostrar modelo
    print("Creando modelo CNN ligero...")
    model = create_cnn_model()
    model = compile_model(model)
    
    # Mostrar información
    print_model_info(model)
    
    # Probar predicción con datos dummy
    import numpy as np
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"Forma de salida: {output.shape}")
    print(f"Suma de probabilidades: {output.sum():.4f}")
