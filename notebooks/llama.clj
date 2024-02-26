^:kindly/hide-code
(ns llama
  (:require
   [clojure.string :as str]
   [com.phronemophobic.llama :as llama]
   [com.phronemophobic.llama.raw :as raw]
   [com.phronemophobic.llama.util :as llutil]
   [scicloj.kindly.v4.kind :as kind]
   [scicloj.kindly.v4.api :as kindly]))

;; # LLMs
;; (Adaptado de [Intro to Runnning LLMs locally][intro])
;;
;; [intro]: https://phronmophobic.github.io/llama.clj/notebooks/intro.html
;;
;; ## Modelo

;; Obtenemos el modelo [Llama2][llama2] de [HuggingFace][huggingface].
;;
;; [huggingface]: https://github.com/phronmophobic/llama.clj?tab=readme-ov-file
;; [llama2]: https://llama.meta.com/

(def llama7b-path "models/llama-2-7b-chat.ggmlv3.q4_0.bin")

;; Creamos un "contexto" para las operaciones con el LLM. Este contexto lo pasamos a
;; otras funciones más adelante.

(def llama-context (llama/create-context llama7b-path {:n-gpu-layers 1}))

;; Las interfaces de chat trabajan con texto, pero los LLMs trabajan con
;; **tokens**.
;;
;; Para tener una idea de las diferencias entre tokens y texto, veamos cómo
;; el modelo de chat llama2 7b tokeniza el texto.

(def sentence "One day I would like to code with LLMs")

;; Los tokens son números. Hay menos tokens que palabras en la oración original.

(def tokens
  (llutil/tokenize llama-context sentence))

(count tokens)

(count sentence)

;; Los tokens son más o menos palabras independientes, pero no exáctamente.

(mapv #(raw/llama_token_to_str llama-context %)
      tokens)

^:kindly/hide-code
(defonce previous* (atom nil))

;; https://en.wikipedia.org/wiki/Logit

^:kindly/hide-code
(defn get-logits [ctx s]
  (raw/llama_set_rng_seed ctx 1234)
  (cond

    (string? s)
    (llama/llama-update ctx s 0)

    (vector? s)
    (let [prev @previous*]
      (if (and
           (vector? prev)
           (= prev (butlast s)))
        (llama/llama-update ctx (last s))
        (do
          (llama/llama-update ctx (llama/bos) 0)
          (run! #(llama/llama-update ctx %)
                s)))))
  (reset! previous* s)

  (into [] (llama/get-logits ctx)))

;; ## Predición

;; Aunque dijimos que la única operación básica de los LLMs es calcular
;; probabilidades, eso no es del todo exacto. Los LLMs calculan [logits][logits]
;; que son ligeramente diferentes. Aunque los logits no son en realidad
;; probabilidades, podemos ignorar los detalles excepto para decir que logits
;; más grandes indican mayor probabilidad y logits más pequeños indican menor
;; probabilidad.
;;
;; [logits]: https://en.wikipedia.org/wiki/Logit

(def atitlan-logits
  (get-logits llama-context "Atitlán is a"))

;; El número de logits es 32,000, que es el número de tokens que nuestro modelo
;; puede representar. Cada índice de la matriz es proporcional a la probabilidad
;; de que el token correspondiente sea el siguiente según nuestro LLM.

;; Dado que los números más altos son más probables, veamos cuáles son los 10
;; primeros candidatos:

(def highest-probability-candidates
  (->> atitlan-logits
       ;; keep track of index
       (map-indexed (fn [idx p]
                      [idx p]))
       ;; take the top 10
       (sort-by second >)
       (take 10)
       (map (fn [[idx _p]]
              (llutil/untokenize llama-context [idx])))))

highest-probability-candidates

(def lowest-probability-candidates
  (->> atitlan-logits
       ;; keep track of index]
       (map-indexed (fn [idx p]
                      [idx p]))
       ;; take the bottom 10
       (sort-by second)
       (take 10)
       (map (fn [[idx _p]]
              (llutil/untokenize llama-context [idx])))))

lowest-probability-candidates

;; ## Generando una respuesta
;;
;; Para obtener un texto completo, utilizamos las probabilidades para elegir el
;; siguiente token. Añadimos ese token a nuestro prompt inicial, repitiendo la
;; operación y obteniendo nuevos _logits_. Y así sucesivamente.

;; Una de las decisiones que la mayoría de las API de LLM ocultan es el método
;; para elegir el siguiente token. En principio, podemos elegir cualquier token
;; y continuar (igual que pudimos elegir el prompt inicial). El nombre para
;; elegir el siguiente token utilizando los logits proporcionados por el LLM se
;; llama **muestreo** (sampling).

;; La elección de un método de muestreo es un tema interesante en sí mismo, pero
;; por ahora, vamos a ir con el método más obvio: Elegiremos el token con la
;; mayor probabilidad. A este método se le llama **greedy sampling**.
;;
;; No suele ser el mejor método, pero es fácil de entender y funciona bastante
;; bien.

;; Ya tenemos un plan para generar una respuesta completa:

;; 1. Alimentar nuestro mensaje inicial en nuestro modelo
;; 2. Muestrear el siguiente token usando el muestreo codicioso.
;; 3. Volver al paso 1 con el token muestreado añadido a nuestro prompt anterior.

;; Pero, ¿cómo sabemos cuándo parar? Los LLMs definen un token que llama.cpp
;; llama fin de frase o eos para abreviar (fin de flujo sería un nombre más
;; apropiado, pero bueno). Podemos repetir los pasos #1-3 hasta que el token eos
;; sea el más probable.

;; Una nota importante: los modelos de chat normalmente tienen un formato de
;; aviso. El formato del prompt es un poco arbitrario y diferentes modelos
;; tendrán diferentes formatos. Llama2 espera que la pregunta esté delimitada
;; por `INST`.

(defn llama2-prompt
  "Meant to work with llama-2-7b-chat.ggmlv3.q4_0.bin"
  [prompt]
  (str
   "[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

" prompt " [/INST]
"))

(defonce response-tokens
  (loop [tokens (llutil/tokenize llama-context
                                 (llama2-prompt "Describe three ways programmers use LLMs"))]
    (let [logits (get-logits llama-context tokens)
          ;; greedy sampling
          token (->> logits
                     (map-indexed (fn [idx p]
                                    [idx p]))
                     (apply max-key second)
                     first)]
      (if (= token (llama/eos))
        token
        (recur (conj tokens token))))))

(def response
  (llutil/untokenize llama-context response-tokens))

^:kindly/hide-code
(-> (last (str/split response #"\[/INST\]"))
    kind/md)


;; ## Ventajas y desventajas
;;
;; Ventajas de LLM privados
;;
;; - Privacidad
;; - Modelos más enfocados
;; - Data más reciente
;; - No hay límite de consultas
;; - Gratis
;; - Tenemos control sobre la entrada y la salida
;; - Podemos usar para otras tareas, por ejemplo clasificación
;;
;; Desventajas
;;
;; - Predicciones tienden a ser inferiores para uso general
;; - Requiere experiencia
;;
;; ## Conclusiones
;;
;; - Modelos van a ser "materia prima"
;; - La diferenciación va a estar en quién los puede integrar mejor a productos
;;
;; [code-llama]: https://ai.meta.com/blog/code-llama-large-language-model-coding/
