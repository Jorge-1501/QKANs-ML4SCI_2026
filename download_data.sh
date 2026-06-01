#!/bin/bash

# paths
DATA_PATH="/home/jorgetoral/Documents/GSoC-26/QKANs-ML4SCI_2026/data/"

# URLs específicas (Zenodo requiere las URLs exactas para cada parte)
URL=(
    "https://zenodo.org/records/2603256/files/test.h5?download=1"
    "https://zenodo.org/records/2603256/files/train.h5?download=1"
    # Expandimos los archivos .npz de Quark-Gluon (ejemplo con partes 0 a 4, ajusta según el record de Zenodo)
    "https://zenodo.org/records/19362155/files/QG_jets_fp32_0.npz?download=1"
    "https://zenodo.org/records/19362155/files/QG_jets_fp32_1.npz?download=1"
    "https://zenodo.org/records/19362155/files/QG_jets_fp32_2.npz?download=1"
    "https://archive.ics.uci.edu/static/public/280/higgs.zip"
)

# Asegurar que el directorio exista antes de entrar
if [ ! -d "$DATA_PATH" ]; then
    echo "📁 Creando directorio: $DATA_PATH"
    mkdir -p "$DATA_PATH"
fi

cd "$DATA_PATH" || exit 1

echo "🚀 Iniciando la descarga de bases de datos para QKANs..."
echo "--------------------------------------------------"

for url in "${URL[@]}"; do
        # Limpiar el nombre del archivo (quita el ?download=1 para que no se guarde con ese nombre raro)
        nombre_archivo=$(basename "$url" | cut -d? -f1)
        
        echo "⏳ Descargando: $nombre_archivo..."
        
        # -O fuerza a guardarlo con el nombre limpio en la ruta actual
        wget -q --show-progress -N -O "$nombre_archivo" "$url"

        if [ $? -eq 0 ]; then
            echo "✅ ¡Éxito! Guardado en: $DATA_PATH$nombre_archivo"
        else
            echo "❌ Error al intentar descargar: $url"
        fi
        echo "--------------------------------------------------"
done

echo "🎉 Proceso de automatización finalizado."
