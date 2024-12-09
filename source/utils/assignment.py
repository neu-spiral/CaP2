import numpy as np
from munkres import Munkres, print_matrix
import sys

def compute_cost_matrix(layer_name, layer_weights, partition):
    """
    Computes the cost of processing an output neuron/channel at each partition.

    Args:
        layer_name (str): The name of the layer.
        layer_weights (torch.Tensor): The weight tensor for the layer.
                                      - For Conv2D: (out_channels, in_channels, kernel_height, kernel_width)
                                      - For Linear: (out_features, in_features)
        partition (dict): The partition information of the model, including this information for the layer:
                          - 'num': Number of partitions
                          - 'filter_id': List of filter IDs for each partition
                          - 'channel_id': List of input channel IDs for each partition
                          - 'maps': Communication cost map between partitions

    Returns:
        np.ndarray: A cost matrix of shape (num_neurons, num_partitions), where:
                    - cost_matrix[i, j] = cost of computing output neuron `i` at partition `j`.
    """
    if layer_name not in partition:
        raise ValueError(f"Layer {layer_name} is not in the partition dictionary.")

    # Extract partition details
    num_partitions = partition[layer_name]['num']
    filter_ids = partition[layer_name]['filter_id']  # Output filters/channels per partition
    channel_ids = partition[layer_name]['channel_id']  # Input channels per partition
    maps = partition[layer_name]['maps']  # Communication cost between partitions

    # Get weight details
    weight_np = layer_weights.cpu().detach().numpy()  # Convert weights to NumPy
    shape = weight_np.shape

    # Initialize cost matrix
    num_neurons = shape[0]  # Number of output neurons
    cost_matrix = np.zeros((num_neurons, num_partitions))

    # Check if the layer is convolutional or fully connected
    is_convolutional = len(shape) == 4  # (out_channels, in_channels, kernel_height, kernel_width)

    # Iterate over output neurons and partitions
    for i in range(num_neurons):  # For each output neuron (or filter)
        for j in range(num_partitions):  # For each partition
            # Calculate cost for input channels from other partitions
            for k in range(num_partitions):  # Input channels from partition k
                if k != j:
                    input_indices = channel_ids[k]  # Input channels for partition k

                    if is_convolutional:
                        # For convolutional layers, check if all kernel elements are zero
                        active_weights = weight_np[i, input_indices, :, :]  # Weights for filter `i` and input channels
                        all_zero = np.all(active_weights == 0, axis=(1, 2))  # Check if all kernel elements are zero
                        num_active = np.sum(~all_zero)  # Count active input channels
                    else:
                        # For fully connected layers, check directly
                        active_weights = weight_np[i, input_indices]  # Weights for input neurons
                        all_zero = active_weights == 0  # Check if weights are zero
                        num_active = np.sum(~all_zero)  # Count active input neurons

                    if num_active > 0:
                        # Add the scaled cost based on the number of active channels/filters
                        cost_matrix[i, j] += (maps[j][k] * num_active)

    return cost_matrix

def list_to_partition(assignments, original_partition, previous_partition):
    """
    Translates a list of machine assignments per neuron back into a partition dictionary.
    
    Args:
        assignments (list or np.ndarray): A list of length num_neurons where assignments[i] = j means
                                          that the i-th neuron is assigned to machine j.
        original_partition (dict): The original partition dictionary, used to retrieve:
                                   - maps: Communication cost between partitions
        previous_partition (dict): The previous layer partition dictionary, used to retrieve:
                                   - channel_id: Input channels per partition

    Returns:
        dict: A new partition dictionary reconstructed based on the assignments.
    """
    # Convert assignments to a NumPy array if it's not already
    assignments = np.array(assignments)
    
    # Validate that the number of partitions (machines) is consistent
    num_partitions = original_partition['num']
    if assignments.max() >= num_partitions:
        raise ValueError("Assignments contain a machine index that exceeds the expected number of partitions.")

    # Reconstruct the partition dictionary
    new_partition = {
        'num': num_partitions,
        'filter_id': [],  # Output neurons (filters) per partition
        'channel_id': previous_partition['filter_id'],  # Retain original input channels
        'maps': original_partition['maps'],  # Retain original communication cost map
    }

    # Populate filter_id for each partition by finding which neurons are assigned to it
    for j in range(num_partitions):
        new_partition['filter_id'].append(np.where(assignments == j)[0])

    return new_partition

def add_virtualmachines(c,b=None):
    '''
    c is an array of shape (num of tasks, num of machines)
    b is the capacity of the machines
    creates dulicates of each machine to exress capacity
    '''
    (n,m) = c.shape
    if n<=m:
        return c,b
    
    if (b is None):
        b=n//m
        if b*m<n:
            b=b+1
    c= np.repeat(c, b, axis=1)
    return c,b
        
def computeassignment(c,b=None):
    """
    computes the assignment problem 
    returns list of elements (task, machine, cost)
    """
    (cv,b)=add_virtualmachines(c,b)
    m = Munkres()
    indexes = m.compute(cv)
    output=[]
    for row, column in indexes:
        rc=column // b
        #print("vm= ", column, "m= ",rc)
        value = c[row][rc]
        output.append((row,rc,value))
    return output
        
def update_assignments(model, configs):
    """
    Recomputes the optimal assignment of output neurons to machines for each layer,
    updates the partition dictionary accordingly, and recalculates communication costs.

    Args:
        model (torch.nn.Module): The current model being trained.
        configs (dict): Configuration dictionary containing:
                        - 'partition': current partitioning of model layers as {layer_name: partition_dict}
                        - 'comm_costs': dictionary of communication costs
                        and other training-related configs.
    """

    # Ensure 'input' partition exists
    if 'input' not in configs['partition']:
        raise KeyError("The 'input' partition is missing in configs['partition']. "
                       "Please define configs['partition']['input'] before calling update_assignments.")

    # Extract all layer names with weights from the model's state_dict
    # A layer_name is derived from state_dict keys ending with '.weight', for example 'conv1.weight' -> 'conv1'
    state_dict_keys = model.state_dict().keys()
    weight_layers = [key[:-7] for key in state_dict_keys if key.endswith('.weight')]

    # Intersect with partitions keys to ensure we only handle layers present in both the model and the partition config
    partition_layer_names = set(configs['partition'].keys())
    # Filter weight_layers to keep only those in partition and also skip 'input' since it's not a real layer
    layer_names = [lname for lname in weight_layers if lname in partition_layer_names and lname != 'input']

    # If no layers found, nothing to update
    if not layer_names:
        print("No layers with weights found in both model and partition. Nothing to update.")
        return

    # Iterate over layers in order. We rely on the order from weight_layers which usually matches model definition order.
    for idx, layer_name in enumerate(layer_names):
        # The original partition for this layer is the current partition state before reassigning
        original_partition = configs['partition'][layer_name].copy()

        # Determine the previous partition
        if idx == 0:
            # For the first layer, use 'input' partition as previous_partition
            previous_partition = configs['partition']['input']
        else:
            # Otherwise, use the partition from the previous layer
            prev_layer_name = layer_names[idx - 1]
            previous_partition = configs['partition'][prev_layer_name]

        # Extract current layer weights
        weight_key = layer_name + '.weight'
        state_dict = model.state_dict()
        if weight_key not in state_dict:
            # If layer has no '.weight', skip this layer (unlikely, but a safe check)
            continue
        layer_weights = state_dict[weight_key]

        # Compute the cost matrix for this layer
        cost_matrix = compute_cost_matrix(layer_name, layer_weights, configs['partition'])

        # Solve the assignment problem
        # computeassignment returns a list of tuples (neuron_idx, machine_idx, cost)
        assignment_solution = computeassignment(cost_matrix)

        # Sort assignments by neuron_idx to ensure correct order
        assignment_solution.sort(key=lambda x: x[0])
        assignments = [machine for (_, machine, _) in assignment_solution]

        # Update partition using list_to_partition with the derived original_partition and previous_partition
        new_partition = list_to_partition(
            assignments,
            original_partition,
            previous_partition
        )

        # Update the current partition for this layer
        configs['partition'][layer_name] = new_partition


if __name__ == "__main__":
    """
    testing assignment solutions
    """
    matrix = [[5, 9, 1,10,10,10,10],
              [5, 9, 1,10,10,10,10],
              [5, 9, 1,10,10,10,10],
              [10, 3, 2,10,10,10,10],
              [10, 3, 2,10,10,10,10],
              [10, 3, 2,10,10,10,10],
              [8, 7, 4,10,10,10,10]]
    
    matrix2= [[2,1,0],
              [0,3,2],
              [1,1,0],
              [2,0,5],
              [0,2,2],
              [1,0,1],
              [2,6,0],
              [0,1,2],
              [5,1,0],
              [2,0,2]]
    
    c=np.array(matrix2)
    print(np.repeat(c, 5, axis=1))
    (cv,b)=add_virtualmachines(c)
    print(c.shape,b)
    print(cv.shape,b)
    
    m = Munkres()
    indexes = m.compute(cv)
    print_matrix(c, msg='Lowest cost through this matrix:')
    total = 0
    print(indexes)
    for row, column in indexes:
        rc=column // b
        #print("vm= ", column, "m= ",rc)
        value = c[row][rc]
        total += value
        print(f'({row}, {rc}) -> {value}')
    print(f'total cost: {total}')
    print("test")
    print(computeassignment(c))
    
        