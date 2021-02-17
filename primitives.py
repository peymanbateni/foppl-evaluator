import torch

def add(args):
    return args[0] + args[1]

def subtract(args):
    return args[0] - args[1]

def multiply(args):
    return args[0] * args[1]

def divide(args):
    return args[0] / args[1]

def vector(args):
    all_are_numbers = True
    for item in args:
        if not torch.is_tensor(item):
            all_are_numbers = False
            break
    if all_are_numbers:
        try:
            if (len(args[0].size()) > 0) and (args[0].size(0) > 1):
                tensor_to_return = torch.stack(args)
            else:
                tensor_to_return = torch.Tensor(args)
            return tensor_to_return
        except Exception:
            return args
    else:
        return args

def get(args):
    if type(args[0]) == dict:
        return args[0][args[1].item()]
    else:
        if torch.is_tensor(args[0]):
            if len(args[0].size()) == 0:
                return args[0]
        return args[0][args[1].long().item()]

def put(args):
    if type(args[0]) == dict:
        new_dict = args[0]
        new_dict[args[1].item()] = args[2]
        return new_dict
    else:
        new_list = args[0]
        new_list[args[1].long().item()] = args[2]
        return new_list

def first(args):
    return args[0][0]

def second(args):
    return args[0][1]

def rest(args):
    return args[0][1:]

def last(args):
    return args[0][-1]

def append(args):
    if type(args[0]) == list:
        return args[0].append(args[1])
    else:
        if len(args[0].size()) > len(args[1].size()):
            return torch.cat((args[0], args[1].unsqueeze(0)), dim=0)
        else:
            return torch.cat((args[0], args[1]), dim=0)

def hashmap(args):
    hashmap_to_return = {}
    for index in range(int(len(args) / 2)):
        hashmap_to_return[args[2*index].item()] = args[2*index+1]
    return hashmap_to_return

def less(args):
    return args[0] < args[1]

def greater(args):
    return args[0] > args[1]

def equal(args):
    return args[0] == args[1]

def sqrt(args):
    return torch.sqrt(args[0])

def normal(args):
    return torch.distributions.Normal(args[0], args[1])

def beta(args):
    return torch.distributions.Beta(args[0], args[1])

def exponential(args):
    return torch.distributions.Exponential(args[0])

def uniform(args):
    return torch.distributions.Uniform(args[0], args[1])

def discrete(args):
    return torch.distributions.Categorical(probs=torch.Tensor(args[0]))

def sample(args):
    return args[0].sample()

def mattranspose(args):
    if len(args[0].size()) > 1:
        return args[0].transpose(0,1)
    else:
        return args[0].unsqueeze(1).transpose(0,1)

def mattanh(args):
    return torch.tanh(args[0])

def matadd(args):
    return args[0] + args[1]

def matmul(args):
    if len(args[0].size()) == 1:
        return torch.matmul(args[0].unsqueeze(1), args[1])
    return torch.matmul(args[0], args[1])

def matrepmat(args):
    if len(args[0].size()) > 1:
        repeated = args[0].repeat(args[2].int().item(), args[1].int().item())
    else:
        repeated = args[0].unsqueeze(1).repeat(args[1].int().item(), args[2].int().item())
    return repeated