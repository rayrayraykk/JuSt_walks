# Jump-Stay-random-walks

This is the python3 and jupyter notebook implementation for

**JUST - Are Meta-Paths Necessary? Revisiting Heterogeneous Graph Embeddings** 

Based on *https://github.com/eXascaleInfolab/JUST*

I modified author's code，added the code of Memory Domain they mentioned but not included. Besides,I added Matrix Factorization methods to get embeddings by sampling for:

**Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec**

And evaluation code was given.



## How to use

### Using conda

`conda install networkx`

`conda install gensim`

`conda install numpy`

`conda install scikit-learn`

`conda install fbpca`

### Quick start

For DBLP

```python
python main.py --input Datasets/DBLP/dblp.edgelist --DATA DBLP --dimensions 128 --walk_length 100 --num_walks 10 --window-size 10 --alpha 0.5 --train 1 --memory 2  --output EmbeddingData
```

For Movie

```python
python main.py --input Datasets/Movie/edgelist_actor_actor_movie_movie_director_composer.edgelist --DATA Movie --dimensions 128 --walk_length 100 --num_walks 10 --window-size 10 --alpha 0.5 --train 1 --memory 2  --output EmbeddingData
```

For Foursquare

```python
python main.py --input Datasets/Foursquare/Foursquare.txt --DATA Foursquare --dimensions 128 --walk_length 100 --num_walks 10 --window-size 10 --alpha 0.5 --train 1 --memory 2  --output EmbeddingData
```



### Evaluation

Use **evaluation.ipynb** to get F1 and NMI score.

### Matrix Factorization method

```python
python main.py --input Datasets/DBLP/dblp.edgelist --DATA DBLP --dimensions 128 --walk_length 100 --num_walks 10 --window-size 10 --alpha 0.5 --train 0 --memory 2
```

By using walk path to get 
$$
P(u, v)
$$
Then using SVD to get embedding:


$$
\overrightarrow{S_{u}} \cdot \overrightarrow{t_{v}}=z=\log \left(\frac{|V| \cdot P(u, v)}{k}\right)=\log \left(\frac{\#(w, c)|\mathcal{D}|}{b \#(w) \#(c)}\right)
$$
