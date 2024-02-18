export train,embedding,anchors,epochs,need_training,model_name,trained_on,norm,modelPhase2,iter_update_params,isAssigned_feature,wandb,num_run,initial_type,initial_order,scoring_func,num_perturbation,node_select,node_select_heuristic,pool,update_reward,hops,num_perturbation_inference,cuttype,finetune_epochs, gamma
train = test_data
embedding = given_lipchitz ##coefficents, Lipschitz_rw, Lipschitz_sp
anchors = 7			# default 20
epochs = 3
need_training=True
model_name=graphsage      ## gcn,gat,graphsage
num_layers=2
trained_on=gap
norm=percentile      # MinMax, Standard, Normalizer,percentile, None
modelPhase2 = ModelLinkPred 			## ModelBasic, ModelAtten, ModelLinkPred, ModelLocalLinkPred
iter_update_params = 2   ## for iteratiing rl parameters
isAssigned_feature = false
gpu=2
wandb=true   ## wandb on of true or false
num_run=5 ## for running multiple times in val and test
initial_type=kmeans   ## random,kmeans
scoring_func=mlp     # mlp,	l1, l2, cosine, dot
initial_order=random  ## random, core_value
num_perturbation=1
node_select=false		# true or false
node_select_heuristic=diff_max		# diff, diff_max, diff_max_scale_degree
pool=mean ## mean, max, sum
update_reward=last ## last,best
hops=2  # 2,3,..
num_perturbation_inference=1000
cuttype=normalised  ##normalised,kmin,sparsest
finetune_epochs=0
gamma=0.99
## code for wandb is added to only phase2_rl
phase2:
	python phase2.py --cuttype $(cuttype) --update_reward $(update_reward) --pool $(pool) --wandb $(wandb) --num_perturbation $(num_perturbation) --num_run $(num_run) --initial_order $(initial_order) --initial_type $(initial_type) --train $(train) --gpu $(gpu) --embedding $(embedding) --anchors $(anchors) --epochs $(epochs) --model_name $(model_name) --num_layers $(num_layers) --need_training $(need_training)  --norm $(norm) --modelPhase2 $(modelPhase2) --iter_update_params $(iter_update_params) --isAssigned_feature $(isAssigned_feature) --scoring_func $(scoring_func) --node_select $(node_select) --node_select_heuristic $(node_select_heuristic) --hops $(hops) --gamma $(gamma)

phase2_finetune:
	python phase2_finetune.py --cuttype $(cuttype) --update_reward $(update_reward) --pool $(pool) --wandb $(wandb) --num_perturbation $(num_perturbation) --num_run $(num_run) --initial_order $(initial_order) --initial_type $(initial_type) --train $(train) --gpu $(gpu) --embedding $(embedding) --anchors $(anchors) --epochs $(epochs) --model_name $(model_name) --num_layers $(num_layers) --need_training $(need_training)  --norm $(norm) --modelPhase2 $(modelPhase2) --iter_update_params $(iter_update_params) --isAssigned_feature $(isAssigned_feature) --scoring_func $(scoring_func) --node_select $(node_select) --node_select_heuristic $(node_select_heuristic) --hops $(hops) --finetune_epochs $(finetune_epochs)

phase2_inference: 
	python phase2_inference.py --num_perturbation_inference $(num_perturbation_inference) --update_reward $(update_reward) --pool $(pool) --wandb $(wandb) --num_perturbation $(num_perturbation) --num_run $(num_run) --initial_order $(initial_order) --initial_type $(initial_type) --train $(train) --gpu $(gpu) --embedding $(embedding) --anchors $(anchors) --epochs $(epochs) --model_name $(model_name) --num_layers $(num_layers) --need_training $(need_training)  --norm $(norm) --modelPhase2 $(modelPhase2) --iter_update_params $(iter_update_params) --isAssigned_feature $(isAssigned_feature) --scoring_func $(scoring_func) --node_select $(node_select) --node_select_heuristic $(node_select_heuristic) --hops $(hops)



clean_pkl: 
	find ../data -name \*.pkl -type f -delete