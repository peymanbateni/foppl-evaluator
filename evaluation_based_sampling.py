from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import torch
import pickle
import primitives

function_map = {'+': primitives.add,
                '-': primitives.subtract,
                '*': primitives.multiply,
                '/': primitives.divide,
                '<': primitives.less,
                '>': primitives.greater,
                '=': primitives.equal,
                'sqrt': primitives.sqrt,
                'vector': primitives.vector,
                'get': primitives.get,
                'put': primitives.put,
                'first': primitives.first,
                'rest': primitives.rest,
                'last': primitives.last,
                'append': primitives.append,
                'hash-map': primitives.hashmap,
                'normal': primitives.normal,
                'sample': primitives.sample,
                'beta': primitives.beta,
                'observe': primitives.sample,
                'exponential': primitives.exponential,
                'uniform': primitives.uniform,
                'second': primitives.second,
                'discrete': primitives.discrete,
                'mat-transpose': primitives.mattranspose,
                'mat-add': primitives.matadd,
                'mat-mul': primitives.matmul,
                'mat-repmat': primitives.matrepmat,
                'mat-tanh': primitives.mattanh}

        
def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    l = {}
    for index in range(len(ast) - 1):
        l = evaluate_program_helper(ast[index], None, l)
    result, sigma, l = evaluate_program_helper(ast[-1], None, l)
    return result, sigma

def evaluate_program_helper(ast, sigma, l):
    # case c
    if type(ast) == str:
        return l[ast], sigma, l
    # case v
    elif type(ast) == int or type(ast) == float:
        return torch.Tensor([ast]), sigma, l
    # case let
    elif len(ast) > 1 and ast[0] == 'let':
        val_evaluated = evaluate_program_helper(ast[1][1], sigma, l)
        l.update({ast[1][0]: val_evaluated[0]})
        return evaluate_program_helper(ast[2], sigma, l)
    # case if
    elif len(ast) > 1 and ast[0] == 'if':
        condition_exp = evaluate_program_helper(ast[1], sigma, l)
        if condition_exp[0]:
            return evaluate_program_helper(ast[2], sigma, l)
        else:
            return evaluate_program_helper(ast[3], sigma, l)
    # case defn
    elif len(ast) > 1 and ast[0] == 'defn':
        l.update({ast[1]: (ast[2], ast[3])})
        return l
    # case (e_0 e_1 e_2 ... e_n)
    else:
        function = ast[0]
        arguments = ast[1:]
        processed_arguments = []
        for argument in arguments:
            processed_arguments.append(evaluate_program_helper(argument, sigma, l)[0])
        if function in function_map.keys():
            return function_map[function](processed_arguments), sigma, l
        else:
            function_args_list = l[function][0]
            function_expression = l[function][1]
            function_args_to_exp = {function_args_list[index]: processed_arguments[index] for index in range(len(function_args_list))} 
            l.update(function_args_to_exp)
            return evaluate_program_helper(function_expression, sigma, l)

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)[0]
    


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        print(ast)
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        print(ast)
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
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
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print(ast)
        print('\n\n\nSample of prior of program {}:'.format(i))
        for j in range(1000):
            print("Sample", j)
            samples[i].append(evaluate_program(ast)[0])

with open("evaluation_based_samples.pk", "wb+") as f:
    pickle.dump(samples, f)