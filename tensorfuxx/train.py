from tensorfuxx.methods import *
      
class GradientDescentOptimizer(object):
    def __init__(self, learning_rate = 0.01):
        self.train_step = learning_rate

    def minimize(self, node):
        topo_order = find_topo_sort([node])
        update = []
        for i in init_value:
            if i in topo_order:
                update.append(i)
        grad = gradients(node, update)
        update_nodes = []
        for i in range(len(update)):
            update_nodes.append(assign(update[i], update[i] - (self.train_step * grad[i])))
        return pack(update_nodes,"train")


class AdamOptimizer(object):
    def __init__(self, learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8,use_locking=False,name="Adam"):
        self.train_step = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.use_locking = use_locking
        self.name = name

    def minimize(self, node):
        topo_order = find_topo_sort([node])
        update = []
        for i in init_value:
            if i in topo_order:
                update.append(i)
        grad = gradients(node, update)
        update_nodes = []
        t = constant(0)
        at = assign(t, t + 1)
        beta1 = constant(self.beta1)
        beta2 = constant(self.beta2)
        lrt = self.train_step*sqrt_op(1 - pow_op(beta2,at))/(1 - pow_op(beta1,at))
        m = [];am = [];v = [];av = [];
        for i in range(len(update)):
           m.append(constant(0))
           v.append(constant(0))
        for i in range(len(update)):
            g = grad[i]
            mt = assign(m[i], m[i] * self.beta1 + g * (1 - self.beta1))
            vt = assign(v[i], v[i] * self.beta2 + g * g * (1 - self.beta2))
            newValue = update[i]-lrt * mt/(sqrt_op(vt) + constant(self.epsilon))
            update_nodes.append(assign(update[i],newValue))
        return pack(update_nodes,"train")
