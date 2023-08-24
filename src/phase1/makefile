# export gpu,train,embedding,anchors,epochs,need_training,model,threshold,pos_weight,num_times_samples,num_edges_samples,trained_on,norm,model_name,num_layers

train = test_data
embedding = coefficents ##coefficents, coefficents_percentile, Lipschitz_rw, Lipschitz_sp
anchors = 20			# default 20
epochs = 10
need_training=True
model_name=graphsage
num_layers=2
threshold=0.5
pos_weight=0.3
num_times_samples=50
num_edges_samples=50
trained_on=gap
norm=percentile      # MinMax, Standard, Normalizer, None
gpu=0                # 0,1,2,3
Run:
	python3 main.py --train $(train) --gpu $(gpu) --embedding $(embedding) --anchors $(anchors) --epochs $(epochs) --model_name $(model_name) --num_layers $(num_layers) --need_training $(need_training) --pos_weight $(pos_weight) --threshold $(threshold) --num_edges_samples $(num_edges_samples) --num_times_samples $(num_times_samples) --trained_on $(trained_on) --norm ${norm} 
run_hmetis:
	# for i in {1..3} ; do \
	# done
	./shmetis $(graphfile) $(cuts) 1 || true
distribute_visualise:
	python3 visualize_cuts.py $(graphfolder)
convert:
	python3 convert.py $(input) $(output)

clean_pkl:
	find ../data -name \*.pkl -type f -delete