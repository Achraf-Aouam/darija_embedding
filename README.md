# Darija Word Embeddings Project

This project focuses on learning high-quality word embeddings for Moroccan Darija using a custom-curated corpus. The process involves extensive text preprocessing, subword tokenization using Byte Pair Encoding (BPE), and training Word2Vec (Skip-gram with Negative Sampling) models.

## Project Goal

To create robust and semantically rich vector representations for Darija words and subwords, which can be used for various downstream Natural Language Processing (NLP) tasks such as sentiment analysis, machine translation, text classification, and information retrieval specific to the Darija dialect.

## Corpus

The primary corpus consists of a collection of `.txt` files sourced from various websites, containing a mix of Darija (written in Arabic or Latin/Arabizi script), Standard Arabic, and some French. The total size of the raw text data is approximately 1 GB, containing around 600 million characters.

**Note on Corpus Curation:**
The "good_articles" portion of the initially considered corpus was excluded from the final training set. This decision was made because these articles predominantly contained Modern Standard Arabic (MSA) rather than Moroccan Darija, and the aim was to create embeddings specifically tailored to Darija.

## Project Architecture & Process

The project is structured into three main stages, implemented across different Jupyter Notebooks:

1.  **Data Preprocessing (`prep_ar_custom.ipynb`)**
    *   **Objective:** To clean and normalize the raw text data, transforming it into a suitable format for tokenization and model training. This step is crucial for handling the diverse and often noisy nature of web-scraped Darija.
    *   **Key Steps:**
        1.  **Corpus Loading:** Reading all `.txt` files from the data directory.
        2.  **Initial Cleaning:** Removing noise such as URLs, email addresses, HTML tags, and irrelevant markup.
        3.  **French Line Detection & Filtering:** Identifying and removing lines that are predominantly French, while retaining lines with mixed Darija/French or Arabizi. (Uses `langdetect` library).
        4.  **Arabization:** Converting Darija written in Latin script (Arabizi), including common numeric substitutions (e.g., '3' for 'ع', '7' for 'ح'), into Arabic script.
        5.  **Arabic Text Normalization:** Standardizing Arabic script by:
            *   Removing diacritics (tashkeel).
            *   Normalizing Alef variants (أ, إ, آ, ٱ) to a plain Alef (ا).
            *   Converting Ta Marbuta (ة) to Ha (ه).
            *   Converting Alef Maksura (ى) to Ya (ي).
            *   Normalizing other character variants (e.g., گ to ك).
        6.  **Final Cleaning:** Removing any remaining non-Arabic characters (except whitespace) and further normalizing whitespace.
        7.  **Short Token Filtering (Optional):** Removing tokens shorter than a specified length, unless they are on an allowlist of common short Darija words.
        8.  **Byte Pair Encoding (BPE) Tokenizer Training:**
            *   Training a BPE subword tokenizer on the processed Arabic script text. This helps manage vocabulary size and handle OOV (Out-Of-Vocabulary) words and morphological variations common in Darija.
            *   Key BPE parameters: `vocab_size`, `min_frequency`.
            *   The trained BPE tokenizer model is saved (e.g., `darija_bpe_tokenizer.json`).
        9.  **Applying BPE Tokenization:** Tokenizing the entire processed corpus using the trained BPE tokenizer.
        10. **Saving Processed Corpus:** The final BPE-tokenized corpus is saved as a text file (e.g., `darija_bpe_tokenized_for_w2v.txt`), with each line containing space-separated BPE tokens. This file serves as the input for Word2Vec training.
        11. **Corpus Analysis:** Generating statistics and visualizations for the tokenized corpus (token frequencies, length distributions).

    *   **RAM Requirement Note:** The full data preprocessing pipeline, especially when handling the entire 1GB corpus in memory for certain steps and BPE training, can be memory-intensive. **Observed RAM usage exceeded 120GB.** To replicate these preprocessing results on the full dataset, it is recommended to use a high-RAM environment, such as a **Lightning AI Data Prep instance (or similar cloud compute with >= 256GB RAM)**. For local development or smaller experiments, using a sample of the data (`sample_size_chars` parameter in the pipeline) is advised.

2.  **Word2Vec Model Training (`word2vecV2.ipynb`)**
    *   **Objective:** To learn vector embeddings for the BPE tokens generated in the previous step.
    *   **Method:** Word2Vec, specifically the Skip-gram algorithm with Negative Sampling.
    *   **Input:** The BPE-tokenized corpus file (e.g., `darija_bpe_tokenized_for_w2v.txt`).
    *   **Key Parameters:**
        *   `vector_size`: Dimensionality of the embedding vectors.
        *   `window`: Context window size.
        *   `min_count`: Minimum frequency for a BPE token to be included in the Word2Vec vocabulary.
        *   `sg=1`: Selects the Skip-gram algorithm.
        *   `negative`: Number of negative samples.
        *   `epochs`: Number of training iterations over the corpus.
        *   `workers`: Number of CPU cores for parallelization.
    *   **Output:** The trained Word2Vec model is saved (e.g., `darija_word2vec_bpe_sg_ns.model`).
    *   **Alternative:** This notebook can also be adapted to train Word2Vec on whole words if BPE tokenization is skipped in the preprocessing stage.

3.  **Inference and Similarity Testing (`inferance_BPE.ipynb` or `inferance_NO_BPE.ipynb`)**
    *   **Objective:** To load the trained Word2Vec model and the BPE tokenizer (if applicable) to perform tasks like finding similar words/tokens or using the embeddings for other applications.
    *   **Process:**
        1.  Load the saved Word2Vec model.
        2.  Load the saved BPE tokenizer model (if using BPE embeddings).
        3.  Implement a user prompt to input a Darija word (in Arabic or Latin script) or phrase.
        4.  Apply the same preprocessing steps (cleaning, arabization, normalization) to the user input.
        5.  Tokenize the processed input using the loaded BPE tokenizer (if applicable).
        6.  Use the Word2Vec model's `most_similar()` method to find the top N closest tokens/words in the vocabulary to the input.
        7.  Display the similar tokens/words and their cosine similarity scores.
    *   **Note:** Separate inference notebooks might exist for models trained with BPE (`inferance_BPE.ipynb`) and without BPE (`inferance_NO_BPE.ipynb`), as the input processing and model interaction differ slightly.


## How to Run

1.  **Setup Environment:**
    *   Install Python and the required libraries (e.g., using `pip install -r requirements.txt` if a `requirements.txt` file is provided).
2.  **Prepare Data:**
    *   Place your Darija `.txt` files in the designated corpus directory (e.g., `data/`).
3.  **Run Preprocessing:**
    *   Open and run the cells in `prep_ar_custom.ipynb`.
    *   **Be mindful of RAM requirements if processing the full dataset.** Adjust `sample_size_chars` for local testing or use a high-RAM environment.
    *   This will generate the BPE tokenizer (`.json`) and the final tokenized corpus (`.txt`).
4.  **Train Word2Vec Model:**
    *   Open and run the cells in `word2vecV2.ipynb`.
    *   Ensure the input corpus path in the notebook points to the output of the preprocessing step.
    *   This will generate the Word2Vec model file (`.model`).
5.  **Perform Inference:**
    *   Open and run `inferance_BPE.ipynb` (or the non-BPE version if applicable).
    *   Ensure the paths to the Word2Vec model and BPE tokenizer (if used) are correctly set.
    *   Interact with the prompt to test word similarities.

