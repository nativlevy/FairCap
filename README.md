brew install graphviz

export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"

pip install pygraphviz

pip install -r requirements.txt