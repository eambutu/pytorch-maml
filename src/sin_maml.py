import click
import math
import os, sys
import numpy as np
import random
import inspect
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import SGD, Adam

from layers import linear


def random_sample():
    phase = np.random.uniform(0, math.pi)
    amp = np.random.uniform(0.1, 5.0)
    return phase, amp

def generate_data(phase, amp, num_points):
    def sinfunc(x):
        return amp * np.sin(x + phase)
    func = np.vectorize(sinfunc)
    xs = np.random.uniform(-5.0, 5.0, num_points)
    ys = func(xs)
    xs = np.expand_dims(xs, axis=1)
    ys = np.expand_dims(ys, axis=1)
    return torch.Tensor(xs), torch.Tensor(ys)

def evaluate(net, phase, amp, weights=None, num_points=5):
    inputs, targets = generate_data(phase, amp, num_points)
    outputs = net.net_forward(inputs, weights)
    loss = nn.MSELoss(outputs, targets).sum()
    return loss.data[0]

class SinusoidNet(nn.Module):
    def __init__(self):
        super(SinusoidNet, self).__init__()
        self.add_module('fc1', nn.Linear(1, 40))
        self.add_module('fc2', nn.Linear(40, 40))
        self.add_module('fc3', nn.Linear(40, 1))

    def forward(self, x, weights=None):
        if weights is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        else:
            x = F.relu(linear(x, weights['fc1.weight'], weights['fc1.bias']))
            x = F.relu(linear(x, weights['fc2.weight'], weights['fc2.bias']))
            x = linear(x, weights['fc3.weight'], weights['fc3.bias'])
            return x

    def net_forward(self, x, weights=None):
        # for use when inherited by inner loop network
        self.forward(x, weights)

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

class InnerLoop(SinusoidNet):
    '''
    This module performs the inner loop of MAML
    The forward method updates weights with gradient steps on training data, 
    then computes and returns a meta-gradient w.r.t. validation data
    '''

    def __init__(self, num_updates, step_size, batch_size, meta_batch_size):
        super(InnerLoop, self).__init__()
        # Number of updates to be taken
        self.num_updates = num_updates

        # Step size for the updates
        self.step_size = step_size

        # PER CLASS Batch size for the updates
        self.batch_size = batch_size

        # for loss normalization 
        self.meta_batch_size = meta_batch_size
    

    def net_forward(self, x, weights=None):
        return super(InnerLoop, self).forward(x, weights)

    def forward_pass(self, in_, target, weights=None):
        ''' Run data through net, return loss and output '''
        input_var = torch.autograd.Variable(in_).cuda(async=True)
        target_var = torch.autograd.Variable(target).cuda(async=True)
        # Run the batch through the net, compute loss
        out = self.net_forward(input_var, weights)
        loss = nn.MSELoss(out, target_var)
        return loss, out
    
    def forward(self, phase, amp):
        phase, amp = random_sample()
        ##### Test net before training, should be random accuracy ####
        pre_loss = evaluate(self, phase, amp)
        fast_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())
        for i in range(self.num_updates):
            print("inner step", i)
            in_, target = generate_data(phase, amp, self.batch_size)
            if i==0:
                loss, _ = self.forward_pass(in_, target)
                grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            else:
                loss, _ = self.forward_pass(in_, target, fast_weights)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = OrderedDict((name, param - self.step_size*grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
        ##### Test net after training, should be better than random ####
        post_loss = evaluate(self, phase, amp)
        print("\nPre inner step loss", pre_loss)
        print("\nPost inner step loss", post_loss)
        
        # Compute the meta gradient and return it
        in_, target = generate_data(phase, amp, self.batch_size)
        loss,_ = self.forward_pass(in_, target, fast_weights) 
        loss = loss / self.meta_batch_size # normalize loss
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
        return post_loss, meta_grads


class MetaLearner(object):
    def __init__(self,
                meta_batch_size, 
                meta_step_size, 
                inner_batch_size, 
                inner_step_size,
                num_updates, 
                num_inner_updates):
        super(self.__class__, self).__init__()
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_batch_size = inner_batch_size
        self.inner_step_size = inner_step_size
        self.num_updates = num_updates
        self.num_inner_updates = num_inner_updates
        
        self.net = SinusoidNet()
        self.net.cuda()
        self.fast_net = InnerLoop(num_classes, self.num_inner_updates, self.inner_step_size, self.inner_batch_size, self.meta_batch_size)
        self.fast_net.cuda()
        self.opt = Adam(self.net.parameters(), lr=meta_step_size)
            
    def meta_update(self, ls):
        print("\n Meta update \n")
        phase, amp = random_sample()
        in_, target = generate_data(phase, amp, self.inner_batch_size)
        # We use a dummy forward / backward pass to get the correct grads into self.net
        loss, out = forward_pass(self.net, in_, target)
        # Unpack the list of grad dicts
        gradients = {k: sum(d[k] for d in ls) for k in ls[0].keys()}
        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []
        for (k,v) in self.net.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))
        # Compute grads for current step, replace with summed gradients as defined by hook
        self.opt.zero_grad()
        loss.backward()
        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt.step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()

    def test(self):
        test_net = SinusoidNet()
        mtr_loss = 0.0
        # Select ten tasks randomly from the test set to evaluate on
        for _ in range(10):
            # Make a test net with same parameters as our current net
            test_net.copy_weights(self.net)
            test_net.cuda()
            test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
            # Train on the train examples, using the same number of updates as in training
            phase, amp = random_sample()
            for i in range(self.num_inner_updates):
                in_, target = generate_data(phase, amp, self.inner_batch_size)
                loss, _  = forward_pass(test_net, in_, target)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
            # Evaluate the trained model on train and val examples
            tloss = evaluate(test_net, phase, amp)
            mtr_loss += tloss

        mtr_loss = mtr_loss / 10

        print('-------------------------')
        print('Meta train loss:', mtr_loss)
        print('-------------------------')
        del test_net
        return mtr_loss

    def _train(self, exp):
        ''' debugging function: learn two tasks '''
        for it in range(self.num_updates):
            grads = []
            # Not sure what to do with this function
            for task in [task1, task2]:
                # Make sure fast net always starts with base weights
                self.fast_net.copy_weights(self.net)
                _, g = self.fast_net.forward(task)
                grads.append(g)
            self.meta_update(grads)
            
    def train(self, exp):
        tr_loss = []
        mtr_loss = []
        for it in range(self.num_updates):
            # Evaluate on test tasks
            mt_loss = self.test()
            mtr_loss.append(mt_loss)
            # Collect a meta batch update
            grads = []
            tloss = 0.0
            for i in range(self.meta_batch_size):
                self.fast_net.copy_weights(self.net)
                trl, g = self.fast_net.forward(task)
                grads.append(g)
                tloss += trl

            # Perform the meta update
            print('Meta update', it)
            self.meta_update(task, grads)

            # Save a model snapshot every now and then
            if it % 500 == 0:
                torch.save(self.net.state_dict(), '../output/{}/train_iter_{}.pth'.format(exp, it))

            # Save stuff
            tr_loss.append(tloss / self.meta_batch_size)

            np.save('../output/{}/tr_loss.npy'.format(exp), np.array(tr_loss))
            np.save('../output/{}/meta_tr_loss.npy'.format(exp), np.array(mtr_loss))

@click.command()
@click.argument('exp')
@click.option('--batch', type=int)
@click.option('--m_batch', type=int)
@click.option('--num_updates', type=int)
@click.option('--num_inner_updates', type=int)
@click.option('--lr',type=str)
@click.option('--meta_lr', type=str)
@click.option('--gpu', default=0)
def main(exp, dataset, batch, m_batch, num_updates, num_inner_updates, lr, meta_lr, gpu):
    random.seed(1337)
    np.random.seed(1337)
    # Print all the args for logging purposes
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for arg in args:
        print(arg, values[arg])

    # make output dir
    output = '../output/{}'.format(exp)
    try:
        os.makedirs(output)
    except:
        pass
    # Set the gpu
    print('Setting GPU to', str(gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    learner = MetaLearner(m_batch, float(meta_lr), batch, float(lr), num_updates, num_inner_updates)
    learner.train(exp)

if __name__ == '__main__':
    main()

