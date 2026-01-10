"""
Definition of the lightweight CNN architecture for skin lesion classification.
Inspired by Mamun et al. (2025).

Architecture:
- 3 convolutional blocks (32, 64, 128 filters)
- MaxPooling after each block
- Flatten
- Dense(256, ReLU, Dropout)
- Softmax(output classes)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

from config.config import MODEL_CONFIG


def create_cnn_model(input_shape=None, num_classes=None, dropout_rate=None):
    """
    Create the lightweight CNN model for skin lesion classification.

    Args:
        input_shape (tuple): Input shape (height, width, channels)
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate

    Returns:
        keras.Model: The Keras model (uncompiled)
    """
    # Use configuration values if not provided
    if input_shape is None:
        input_shape = MODEL_CONFIG['input_shape']
    if num_classes is None:
        num_classes = MODEL_CONFIG['num_classes']
    if dropout_rate is None:
        dropout_rate = MODEL_CONFIG['dropout_rate']
    
    # Input
    inputs = layers.Input(shape=input_shape, name='input_images')
    x = inputs
    
    # Convolutional Block 1: 32 filters
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
    
    # Convolutional Block 2: 64 filters
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
    
    # Convolutional Block 3: 128 filters
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
    
    # Store for Grad-CAM (last convolutional layer)
    last_conv_layer = x
    
    # Flatten
    x = layers.Flatten(name='flatten')(x)
    
    # Dense layer with dropout
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=l2(0.001),
        name='dense1'
    )(x)
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    
    # Output layer (softmax)
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='LightCNN_SkinCancer')
    
    return model


def compile_model(model, learning_rate=None, loss_function=None):
    """
    Compile the model with optimizer, loss and metrics.

    Args:
        model (keras.Model): Model to compile
        learning_rate (float): Learning rate
        loss_function (str): Loss function identifier

    Returns:
        keras.Model: Compiled model
    """
    from config.config import TRAINING_CONFIG
    
    # Use configuration values if not provided
    if learning_rate is None:
        learning_rate = TRAINING_CONFIG['learning_rate']
    if loss_function is None:
        loss_function = TRAINING_CONFIG['loss_function']
    
    # Configure optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=TRAINING_CONFIG['optimizer_params']['beta_1'],
        beta_2=TRAINING_CONFIG['optimizer_params']['beta_2'],
        epsilon=TRAINING_CONFIG['optimizer_params']['epsilon']
    )
    
    # Configure loss function
    if loss_function == 'focal_loss':
        # TODO: Implement focal loss for class imbalance
        loss = keras.losses.CategoricalCrossentropy()
    else:
        loss = keras.losses.CategoricalCrossentropy()
    
    # Metrics
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
    Get a string summary of the model.

    Args:
        model (keras.Model): The model

    Returns:
        str: Model summary
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def count_parameters(model):
    """
    Count total and trainable parameters of the model.

    Args:
        model (keras.Model): The model

    Returns:
        dict: Dictionary with parameter counts
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
    Create focal loss function to handle class imbalance.

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Balance factor
        gamma (float): Focusing parameter

    Returns:
        function: Focal loss function
    """
    def focal_loss(y_true, y_pred):
        """
        Compute focal loss.

        Args:
            y_true: True labels (one-hot)
            y_pred: Model predictions

        Returns:
            tensor: Calculated loss
        """
        # TODO: Implement focal loss
        # epsilon for numerical stability
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
    Get the name of the last convolutional layer (for Grad-CAM).

    Args:
        model (keras.Model): The model

    Returns:
        str: Name of the last convolutional layer or None
    """
    conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
    if conv_layers:
        return conv_layers[-1]
    return None


# ==================== UTILITY FUNCTIONS ====================

def save_model_architecture(model, filepath='model_architecture.png'):
    """
    Save the model architecture as an image.

    Args:
        model (keras.Model): The model
        filepath (str): Path to save the image
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
        print(f"Architecture saved to: {filepath}")
    except Exception as e:
        print(f"Error saving architecture: {e}")


def print_model_info(model):
    """
    Print detailed model information.

    Args:
        model (keras.Model): The model
    """
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    
    # Resumen
    model.summary()
    
    # Parameters
    params = count_parameters(model)
    print("\nPARAMETERS:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")

    # Last convolutional layer
    last_conv = get_last_conv_layer_name(model)
    print(f"\nLast convolutional layer (Grad-CAM): {last_conv}")
    
    print("=" * 60 + "\n")


# ==================== TESTING ====================

if __name__ == '__main__':
    # Create and display model
    print("Creating light CNN model...")
    model = create_cnn_model()
    model = compile_model(model)
    
    # Mostrar información
    print_model_info(model)
    
    # Test prediction with dummy data
    import numpy as np
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"Output shape: {output.shape}")
    print(f"Sum of probabilities: {output.sum():.4f}")
