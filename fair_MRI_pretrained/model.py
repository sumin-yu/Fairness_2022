import torch
from torch import nn
from networks import resnet


def generate_model(args):
    # assert opt.model in [
    #     'resnet'
    # ]

    # if opt.model == 'resnet':
    #     assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
    resnet_shortcut = 'B'
    if args.network == 'resnet10':
        model = resnet.resnet10(
            # sample_input_W=opt.input_W,
            # sample_input_H=opt.input_H,
            # sample_input_D=opt.input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=args.no_cuda
            # num_seg_classes=opt.n_seg_classes
            )
    elif args.network == 'resnet18':
        model = resnet.resnet18(
            # sample_input_W=opt.input_W,
            # sample_input_H=opt.input_H,
            # sample_input_D=opt.input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=args.no_cuda
            # num_seg_classes=opt.n_seg_classes
            )
    elif args.network == 'resnet34':
        model = resnet.resnet34(
            # sample_input_W=opt.input_W,
            # sample_input_H=opt.input_H,
            # sample_input_D=opt.input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=args.no_cuda
            # num_seg_classes=opt.n_seg_classes
            )
    elif args.network == 'resnet50':
        model = resnet.resnet50(
            # sample_input_W=opt.input_W,
            # sample_input_H=opt.input_H,
            # sample_input_D=opt.input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=args.no_cuda,
            # num_seg_classes=opt.n_seg_classes
            )
    elif args.network == 'resnet101':
        model = resnet.resnet101(
            # sample_input_W=opt.input_W,
            # sample_input_H=opt.input_H,
            # sample_input_D=opt.input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=args.no_cuda
            # num_seg_classes=opt.n_seg_classes
            )
    elif args.network == 'resnet152':
        model = resnet.resnet152(
            # sample_input_W=opt.input_W,
            # sample_input_H=opt.input_H,
            # sample_input_D=opt.input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=args.no_cuda
            # num_seg_classes=opt.n_seg_classes
            )
    elif args.network == 'resnet200':
        model = resnet.resnet200(
            # sample_input_W=opt.input_W,
            # sample_input_H=opt.input_H,
            # sample_input_D=opt.input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=args.no_cuda
            # num_seg_classes=opt.n_seg_classes
            )
    
    # if not args.no_cuda:
    #     if len(opt.gpu_id) > 1:
    #         model = model.cuda() 
    #         model = nn.DataParallel(model, device_ids=opt.gpu_id)
    #         net_dict = model.state_dict() 
    #     else:
    #         import os
    #         os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
    #         model = model.cuda() 
    #         model = nn.DataParallel(model, device_ids=None)
    #         net_dict = model.state_dict()
    # else:
    if args.cuda:
        model = model.cuda()
        net_dict = model.state_dict()
    
    # load pretrain
    if args.mode != 'test' and args.pretrain_path:
        print ('loading pretrained model {}'.format(args.pretrain_path))
        pretrain = torch.load(args.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = [] 
        for pname, p in model.named_parameters():
            for layer_name in args.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()
