"""
Script de configuración inicial del proyecto.
Crea directorios necesarios y valida la configuración.

Uso:
    python setup_project.py
"""

import os
from pathlib import Path
import sys

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import (
    LOGGING_CONFIG, 
    DATA_CONFIG,
    SECURITY_CONFIG,
    print_config_summary
)


def create_directory_structure():
    """Crea la estructura de directorios necesaria."""
    print("=" * 60)
    print("CREANDO ESTRUCTURA DE DIRECTORIOS")
    print("=" * 60)
    
    # Directorios principales
    directories = [
        # Logs y resultados
        LOGGING_CONFIG['log_dir'],
        LOGGING_CONFIG['tensorboard_dir'],
        LOGGING_CONFIG['model_checkpoint_dir'],
        LOGGING_CONFIG['results_dir'],
        
        # Datos
        DATA_CONFIG['data_root'],
        
        # Seguridad (certificados)
        Path(SECURITY_CONFIG.get('certificate_path', './certs')).parent,
    ]
    
    # Crear cada directorio
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Creado: {dir_path}")
        else:
            print(f"✓ Ya existe: {dir_path}")
    
    print("\n" + "=" * 60)


def verify_dependencies():
    """Verifica que las dependencias estén instaladas."""
    print("\n" + "=" * 60)
    print("VERIFICANDO DEPENDENCIAS")
    print("=" * 60)
    
    required_packages = [
        ('flwr', 'Flower'),
        ('tensorflow', 'TensorFlow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('cv2', 'OpenCV')
    ]
    
    missing = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name} instalado")
        except ImportError:
            print(f"✗ {name} NO instalado")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Paquetes faltantes: {', '.join(missing)}")
        print("Ejecuta: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ Todas las dependencias están instaladas")
        return True
    
    print("=" * 60)


def check_gpu_availability():
    """Verifica disponibilidad de GPU."""
    print("\n" + "=" * 60)
    print("VERIFICANDO GPU")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ GPUs disponibles: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                
            # Configurar memoria dinámica
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  ✓ Memoria dinámica habilitada para {gpu.name}")
                except RuntimeError as e:
                    print(f"  ⚠ No se pudo configurar memoria dinámica: {e}")
        else:
            print("⚠ No se detectaron GPUs - se usará CPU")
            print("  El entrenamiento será más lento")
    
    except Exception as e:
        print(f"✗ Error verificando GPU: {e}")
    
    print("=" * 60)


def create_dataset_readme():
    """Crea README en carpeta de datasets con instrucciones."""
    datasets_path = Path(DATA_CONFIG['data_root'])
    readme_path = datasets_path / 'README.md'
    
    readme_content = """# Datasets para Federated Learning

## Estructura Requerida

Descarga y organiza los datasets en la siguiente estructura:

```
datasets/
│
├── HAM10000/
│   ├── images/
│   │   ├── ISIC_0024306.jpg
│   │   └── ...
│   └── HAM10000_metadata.csv
│
├── ISIC2018/
│   ├── ISIC2018_Task3_Training_Input/
│   │   ├── ISIC_0000000.jpg
│   │   └── ...
│   └── ISIC2018_Task3_Training_GroundTruth.csv
│
├── ISIC2019/
│   ├── ISIC_2019_Training_Input/
│   │   ├── ISIC_0000000.jpg
│   │   └── ...
│   └── ISIC_2019_Training_GroundTruth.csv
│
├── ISIC2020/
│   ├── train/
│   │   ├── ISIC_0000000.jpg
│   │   └── ...
│   └── train.csv
│
└── PH2/
    ├── images/
    │   └── ...
    └── PH2_dataset.csv
```

## Fuentes de Datos

### HAM10000 (Primary Node)
- **Fuente**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- **Clases**: 7 tipos de lesiones cutáneas
- **Tamaño**: ~10,015 imágenes

### ISIC 2018 Challenge (Node 1)
- **Fuente**: https://challenge.isic-archive.com/data/#2018
- **Task 3**: Lesion Diagnosis
- **Clases**: 7 tipos de lesiones

### ISIC 2019 Challenge (Node 3)
- **Fuente**: https://challenge.isic-archive.com/data/#2019
- **Clases**: 8 tipos (9 con "Unknown")
- **Tamaño**: ~25,331 imágenes

### ISIC 2020 Challenge (Node 2)
- **Fuente**: https://challenge.isic-archive.com/data/#2020
- **Enfoque**: Binary classification (malignant/benign)
- **Tamaño**: ~33,126 imágenes

### PH2 (External Validation)
- **Fuente**: https://www.fc.up.pt/addi/ph2%20database.html
- **Uso**: Solo para validación externa final
- **Tamaño**: 200 imágenes dermoscópicas

## Notas Importantes

1. **Preprocesamiento**: Todas las imágenes se redimensionarán a 224×224
2. **Balance**: Se aplicarán class weights y data augmentation
3. **Splits**: 70% train, 15% val, 15% test
4. **IID/Non-IID**: Configurable para simulaciones realistas

## Licencias

Cada dataset tiene su propia licencia. Verifica los términos de uso:
- HAM10000: CC BY-NC 4.0
- ISIC Challenges: Terms available at isic-archive.com
- PH2: Academic use only
"""
    
    readme_path.write_text(readme_content, encoding='utf-8')
    print(f"\n✓ README de datasets creado: {readme_path}")


def print_next_steps():
    """Imprime los siguientes pasos."""
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN COMPLETADA")
    print("=" * 60)
    print("\n📋 PRÓXIMOS PASOS:\n")
    print("1. Descargar datasets (ver datasets/README.md)")
    print("2. Verificar estructura de carpetas")
    print("3. Probar carga de datos:")
    print("   python -m data.data_loader")
    print("\n4. Iniciar servidor:")
    print("   python main_server.py --strategy FedAvg --rounds 50")
    print("\n5. Iniciar clientes (en terminales separadas):")
    print("   python main_client.py --node-id 0 --dataset HAM10000")
    print("   python main_client.py --node-id 1 --dataset ISIC2018")
    print("   python main_client.py --node-id 2 --dataset ISIC2020")
    print("   python main_client.py --node-id 3 --dataset ISIC2019")
    print("\n6. Monitorear con TensorBoard:")
    print("   tensorboard --logdir=logs/tensorboard")
    print("\n" + "=" * 60)


def main():
    """Ejecuta la configuración inicial."""
    print("\n" + "🚀 " * 20)
    print("CONFIGURACIÓN INICIAL DEL PROYECTO")
    print("Federated Learning - Clasificación de Cáncer de Piel")
    print("🚀 " * 20 + "\n")
    
    # Mostrar configuración
    print_config_summary()
    
    # Crear estructura de directorios
    create_directory_structure()
    
    # Crear README de datasets
    create_dataset_readme()
    
    # Verificar dependencias
    deps_ok = verify_dependencies()
    
    # Verificar GPU
    check_gpu_availability()
    
    # Siguiente pasos
    print_next_steps()
    
    if not deps_ok:
        print("\n⚠ ATENCIÓN: Instala las dependencias faltantes antes de continuar")
        return 1
    
    print("\n✅ ¡Sistema configurado correctamente!\n")
    return 0


if __name__ == '__main__':
    exit(main())
