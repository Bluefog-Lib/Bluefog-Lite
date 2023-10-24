bflrun -np 4 python example/torch_mnist.py \
        --epochs 100 \

# bflrun -np 4 python example/torch_avg.py \
#         --consensus-method neighbor_allreduce_nonblocking

# bflrun -np 4 python example/torch_avg.py \
#         --consensus-method neighbor_allreduce

# bflrun -np 4 python example/torch_avg.py \
#         --consensus-method allreduce_nonblocking

# bflrun -np 4 python example/torch_avg.py \
#         --consensus-method allreduce




