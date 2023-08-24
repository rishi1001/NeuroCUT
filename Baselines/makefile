train = cora
test  = cora
embedding = lipschitz
anchors = 100
cuts = 5
epochs = 1000
components = 100
training=true
b=1
Gap:
	$(MAKE) -C src/GAP-Code train=$(train) test=$(test) embedding=$(embedding) anchors=$(anchors) cuts=$(cuts) epochs=$(epochs) components=$(components) training=$(training)
non_nueral:
	python3 convert.py data/sample_graph/test_set/1/