export train,test, embedding, anchors, epochs, components, training

all:
	python3 TrialModel.py --train $(train) --test $(test) --embedding $(embedding) --cuts $(cuts) --anchors $(anchors) --epochs $(epochs) --components $(components) --training $(training)

normal:
	python3 TrialModel.py --train citeseer --test cora --embedding lipschitz --cuts 5 --anchors 100 --epochs 1000 --training
