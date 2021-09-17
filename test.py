import numpy as np
import argparse
import torch
from network import Bi3DOF, Encoder, Decoder
from datasets import seed, Bi3DOFDataset
'''
	Inputs are test clips are hdf5 files prodiced byfeature_abstraction.py.
	Place all test clips in folder data/nuscenes-v1.0-mini.test.

	Outputs are scores for OoD detection.
'''
def load_model(test_config): 
	parser = argparse.ArgumentParser() 	   
	args = parser.parse_args()
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Default values 
	args.training = False
	args.kl_weight = 1
	args.group = 2	
	args.nz = 12	 
	args.nd = 6
	args.mu1, args.mu2 =0 , 0
	# test config
	args.latentprior = test_config["network"]
	args.var1, args.var2 = 1, 1

	# Input dimension as in # [grp,nd,h,w]
	args.input_size = [args.group, args.nd, 120,160]   
	args.transform_size = [113,152]   

	# Assifn test clips
	args.data_file = test_config["test_clips"]
	# Sequentially advance 1 frame per step 
	args.n_seq = test_config["frames_per_clip"] - args.nd

	# Load  weights  
	encoder = Encoder(args)
	decoder = Decoder(args)
	model = Bi3DOF(encoder, decoder, args).to(args.device)   
	model.load_state_dict(torch.load(test_config["model_file"], map_location = args.device))
	model.eval()
	return model, args

def compute_score(model, args):
	args.batch_size = 1
	testset = Bi3DOFDataset(args)    
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
	d_horizontal = []
	d_vertical = []
	model.eval()
	with torch.no_grad():
		for _, (batch_data) in enumerate(test_loader): 
			data = batch_data.to(args.device)
			(b1,b2,g,d,h,w) = data.shape  
			data = data.view((b1*b2,g,d,h,w))
			_, (d_grp1, d_grp2) = model.encode(data)
			for i in range(len(d_grp1)):
				d_horizontal.append(d_grp1[i].cpu().numpy()) 
				d_vertical.append(d_grp2[i].cpu().numpy())
	return d_horizontal, d_vertical

seed()
bi3dof_simple = {
    "model_file" : "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
    "network" : "simple",   
    "test_clips": "data/nuscenes-v1.0-mini.test",      
    "frames_per_clip": 48
}
model, args = load_model(bi3dof_simple)

h,v = compute_score(model, args)
for i in range(len(h)): print("{:.6f}".format(h[i]+v[i]))
print("\n")