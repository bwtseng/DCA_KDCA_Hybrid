import argparse
import os 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attack')
	parser.add_argument('--train',type=bool,default=False,help='Training')
	parser.add_argument('--train_1',type=bool,default=False,help='Training')
	parser.add_argument('--ori_dim', type=int, default=23, help='Orignal dimension')
	parser.add_argument('--com_dim', type=int, default=11, help='Compressive dimension')
	parser.add_argument('--ridge', type=float, default=0.01, help='Ridge term for regression')
	parser.add_argument('--batch_size', type=int, default=1000, help='Training batch size')
	parser.add_argument('--epoch', type=int, default=200, help='Training epcohs')
	parser.add_argument('--gamma', type=float, default=0.001, help='Gamma of kernel.')
	parser.add_argument('--kernel', type=str, default='rbf', help='rbf polynomial laplacian linear')
	args = parser.parse_args()
	print(args)

	if args.train: 
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		import dca
		model = dca.dca(args)
		model.train()

	elif args.train_1 :
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import dca
		model = dca.dca(args)
		model.svm_pred()


	else:
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="1"
		import dca
		model = dca.dca(args)
		model.train()