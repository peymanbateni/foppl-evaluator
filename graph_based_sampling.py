import torch
import torch.distributions as dist

from daphne import daphne

import primitivesgb
from tests import is_tol, run_prob_test,load_truth
import pickle

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'+': primitivesgb.add,
                '-': primitivesgb.subtract,
                '*': primitivesgb.multiply,
                '/': primitivesgb.divide,
                '<': primitivesgb.less,
                '>': primitivesgb.greater,
                '=': primitivesgb.equal,
                'sqrt': primitivesgb.sqrt,
                'vector': primitivesgb.vector,
                'get': primitivesgb.get,
                'put': primitivesgb.put,
                'first': primitivesgb.first,
                'rest': primitivesgb.rest,
                'last': primitivesgb.last,
                'append': primitivesgb.append,
                'hash-map': primitivesgb.hashmap,
                'normal': primitivesgb.normal,
                'sample*': primitivesgb.sample,
                'beta': primitivesgb.beta,
                'observe': primitivesgb.sample,
                'exponential': primitivesgb.exponential,
                'uniform': primitivesgb.uniform,
                'second': primitivesgb.second,
                'discrete': primitivesgb.discrete,
                'mat-transpose': primitivesgb.mattranspose,
                'mat-add': primitivesgb.matadd,
                'mat-mul': primitivesgb.matmul,
                'mat-repmat': primitivesgb.matrepmat,
                'mat-tanh': primitivesgb.mattanh,
                'if': primitivesgb.iffunction}


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    elif type(exp) is str:
        return env[exp]
    else:
        raise("Expression type unknown.", exp)


def var_of_concern(var, var_list):
    for item in var_list:
        if type(item) is list:
            if var_of_concern(var, item):
                return True
        elif item == var:
            return True
    return False

def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    graph_info = graph[1]

    for var in graph_info['V']:
        if type(graph[-1]) is list:
            if not var_of_concern(var, graph[-1]):
                continue
        else:
            if not var == graph[-1]:
                continue
        if 'observe' in var:
            vals_to_return.append(None)

        dependent_vars = ['vector']
        for tmp_var in graph_info['A'].keys():
            if tmp_var in ['uniform']:
                continue
            if var in graph_info['A'][tmp_var]:
                dependent_vars.append(tmp_var)
        
        if len(dependent_vars) > 1:
            tmp_graph = graph[:-1] + dependent_vars
            dependencies_vals = sample_from_joint(tmp_graph)
            for index, dep_var in enumerate(dependent_vars[1:]):
                if type(dependencies_vals) is not list:
                    env[dep_var] = dependencies_vals
                else:
                    env[dep_var] = dependencies_vals[index]
                

        var_evaluted = deterministic_eval(graph_info['P'][var])
        env[var] = var_evaluted

    return deterministic_eval(graph[-1])


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:
def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        print(graph)
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():

    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        print(graph)
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        #stream = get_stream(graph[-1])
        stream = get_stream(graph)

        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    run_deterministic_tests()
    run_probabilistic_tests()

    samples = {}
    for i in range(1,5):
        samples[i] = []
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        for j in range(1000):
            print("Sample", j)
            samples[i].append(sample_from_joint(graph))

with open("graph_based_samples.pk", "wb+") as f:
    pickle.dump(samples, f)