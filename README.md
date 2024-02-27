# LLama

[nb]: https://kpassapk.github.io/llama.clj/llama.html

Explorando [llama.clj][llama-clj]. Ver [Notebook][nb].

(Disclaimer: Probado solamente en una Mac Apple Silicon.)

[llama-clj]: https://github.com/phronmophobic/llama.clj/

## Requerimientos

- `cmake`

## Setup

1. Descargar `llama.cpp`
2. Correr estos comandos

```
cd {{ directorio donde descargue llama.cpp }}
mkdir build
cmake -DBUILD_SHARED_LIBS=ON .. 
```

2. Modificar `deps.edn para apuntar al directorio de 'build'`

3. "Jack-in" con [calva][calva] o similar, escogiendo `tools.deps` y el alias `cider` en los prompts

4. Ejecutar  `notebooks/llama.clj`y modificar

[calva]: https://calva.io/

``` 4d
clj -M:cider
```
