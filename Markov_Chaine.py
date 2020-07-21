import random as rd
import pickle as pkl

class MC:
    """
    The markov chaine object
    """

    def __init__(self, tokens, begin_token, ending_token):
        """
        tokens: the list of all tokens cuse in the markov chaine
                each token should be an invariant type of data (tuples, int, strings etc....)
        begin_token: the token who mark the begining of a chaine of tokens
        ending_token: the token who mark the end of a chaine of tokens
        """

        # verification
        if begin_token not in tokens:
            raise ValueError("The begin_token must be contain in the tokens list")

        if ending_token not in tokens: 
            raise ValueError("The ending_token must be contain in the tokens list")

        # create basic attribut
        self._trained = False
        self.nodes = {}
        self.begin_token = begin_token
        self.ending_token = ending_token
        self.tokens = tokens

        # create the nodes dict
        for token in tokens:
            self.nodes[token] = Node(token)

    def is_trained(self):
        """
        check if the chaine is trained
        
        return:
            - trained (bool): True if the chaine is trained else False 
        """
        return self._trained

    def train(self, training_data):
        """
        traine the markov chaine based on data

        parametter:
            - training_data [[token,],]: a list of list of tokens contain in the token list
        """

        # check if the chaine isn't already trained
        if self._trained == True:
            raise RuntimeError("The chaine is already trained")

        # the training loop
        for i, token_chaine in enumerate(training_data):
            print("\rtraining: %s / %s" %(i+1, len(training_data)), end="")

            # train on each token chaine
            for j, token in enumerate(token_chaine):

                # check data integrity
                if token not in self.tokens:
                    raise ValueError("Token chaine %s token %s : the token %s should only be contain in the tokens list use during initalisation" %(i+1, j+1, token))

                if j == 0 and token != self.begin_token:
                    raise ValueError("Token chaine %s: the begin of each token chaine should be equal to the begin_token: %s != %s" %(i+1, token, self.begin_token))

                if j+1 == len(token_chaine) and token != self.ending_token:
                     raise ValueError("Token chaine %s: the endin of each token chaine should be equal to the end_token: %s != %s" %(i+1, token, self.ending_token))

                if j != 0 and token == self.begin_token:
                    raise ValueError("Token chaine %s token %s : the begin_token %s should only be use at the begin of the token chaine" %(i+1, j+1, self.begin_token))

                if j+1 != len(token_chaine) and token == self.ending_token:
                    raise ValueError("Token chaine %s token %s : the ending_token %s should only be use at the end of the token chaine" %(i+1, j+1, self.ending_token))

                # traine the chaine
                if j != 0:
                    # get the usefull nodes
                    prev_node = self.nodes[token_chaine[j-1]]
                    current_node = self.nodes[token]

                    # add to the previous node the current node
                    prev_node.count_transition(current_node)

        print("\n")

        # update the nodes for the prediction
        for i, node in enumerate(self.nodes):
            print("\rupdating nodes (%s/%s)" %(i+1, len(self.nodes)), end="")
            self.nodes[node].update()

        # end the training
        print("\nTraining finished\n")
        self._trained = True

    def predict(self, max_tokens=10_000):
        """
        Use the contructed model to predict a list of tokens

        parametter:
            max_tokens (int): the number of tokens max if we don't reach the end token

        return:
            prediction (list of tokens): the prediction creat by the model
        """
        prediction = [self.begin_token,]
        while True:
            # predict the current token
            previous_node = self.nodes[prediction[-1]]
            current_node = previous_node.next_node()
            prediction.append(current_node)

            # return
            if len(prediction) == max_tokens or self.ending_token == prediction[-1]:
                return prediction

class Node:
    """
    A node of the markof chaine
    """

    def __init__(self, token):
        """
        token: the token associate to the node
        """

        self.passage = 0
        self.output_nodes_probs = {} 
        self.output_nodes_trained = []
        self.token = token

    def count_transition(self, next_node):
        """
        count the transition form this node to the next_node

        parametter: 
            next_node (Node): the next node in the transition
        """

        self.passage += 1 

        if next_node.token not in self.output_nodes_probs:
            # add the node
            self.output_nodes_probs[next_node.token] = 1
        else:
            # update the transitions
            self.output_nodes_probs[next_node.token] += 1

    def update(self):
        """
        update the node afer training for predictions
        """

        # create the node list
        for node in self.output_nodes_probs:
            node_dic = {"node": node, "value": self.output_nodes_probs[node]}
            self.output_nodes_trained.append(node_dic)

        # sorte the list
        self.output_nodes_trained = sorted(self.output_nodes_trained, key= lambda k : k["value"])

        # update the value of each node
        total = 0
        for dic in self.output_nodes_trained:
            total += dic["value"] 
            dic["value"] = total / self.passage


    def next_node(self):
        """
        Select random the next node and return it

        return:
            selected (Node) the randomly selected node
        """
        number = rd.random()
        for dic in self.output_nodes_trained:
            if dic["value"] > number:
                return dic["node"]


def save_chaine(chaine, path):
    """
    Save a markov chaine

    parametters:
        - chaine (MC): the markov chaine object
        - path (str): the path were the markov chaine will be saved
    """
    with open(path, 'wb') as fp:
        pkl.dump(chaine, fp)

def load_chaine(path):
    """
    Load a chaine save on the path

    parametter:
        - path (str): the path were the chaine is saved
    
    return:
        - chaine (MC): the chaine loaded
    """
    with open(path, "rb") as fp:
        chaine = pkl.load(fp)

    return chaine

if __name__ == "__main__":
    # test
    tokens = ["<beg>", "a", "e", "i", "o", "u", "<end>"]
    begin_token = "<beg>"
    ending_token = "<end>"
    training_data = [ [begin_token,] + [ rd.choice(["a", "e", "i", "o", "u"]) for j in range(25) ] + [ending_token,] for i in range(1000)]


    chaine = MC(tokens, begin_token, ending_token) 


    print("Chaine data dict:")
    print(chaine.nodes)
    print("\n")

    print("Chaine training state:")
    print(chaine.is_trained())
    print("\n")

    print("Train the data")
    chaine.train(training_data)
    print("\n")

    print("The nodes + the passages after training")
    for node in chaine.nodes:
        node = chaine.nodes[node]
        print(node.token, node.passage)
        for other_node in node.output_nodes_probs:
            print("\t", other_node, node.output_nodes_probs[other_node])

        print("")

    print("Some predictions")
    for i in range(10):
        print(chaine.predict())

    save_chaine(chaine, "chaine.save")
    chaine = load_chaine("chaine.save")

    print("Some predictions after loading and saving")
    for i in range(10):
        print(chaine.predict())


