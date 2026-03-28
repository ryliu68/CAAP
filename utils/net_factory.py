# ROOT/utils/net_factory.py
from models import *

def get_net(args):

    if args.net == "vgg16":
        net = vgg16_bn(num_class=args.num_class)
    elif args.net == "resnet18":
        net = resnet18(num_class=args.num_class)   
    elif args.net == "mobilenetv2":
        net = mobilenetv2(num_class=args.num_class)
    elif args.net == "shufflenetv2":
        net = shufflenetv2(num_class=args.num_class)
    elif args.net == "compnet":
        net = compnet(num_class=args.num_class)
    elif args.net == "ccnet":
        net = ccnet(num_class=args.num_class,weight=0.8)
    elif args.net == "co3net":   
        net = co3net(num_class=args.num_class)
    else:
        print(args.net)
        raise ValueError("Not implemented")
    
    return net
    