#include "neural_net.h"
ThreadPool layer::threads;
neuron::neuron(){}
neuron::neuron(size_t input_size, activation func): weights(input_size),function(func)
{
	bias.random_init(); 
	weights.random_init(); 
	bias.requires_grad();
	weights.requires_grad();
}
void neuron::set_input(value_array & in)
{
	input = std::move(in.return_copy()); 
}
void neuron::forward()
{
	weights_input = std::move(input*weights);
	sum = std::move(weights_input.sum());
	sumwbias = std::move(sum + bias);
	switch(function){
	case TANH: 
		final = std::move(sumwbias.tanh());
		return; 
	case ELU: 
		final = std::move(sumwbias.elu());
		return; 
	case NONE: 
		break;
	}
}
value& neuron::neuron_output()
{
if (function == NONE)
	return sumwbias; 
return final;
}
layer::layer(){}

layer::layer(size_t input_size, size_t amount_of_neurons, activation func):
	function(func),
	final(amount_of_neurons)
{
	m_neurons.reserve(amount_of_neurons); 
	for(size_t i=0; i< amount_of_neurons; i++)
		m_neurons.emplace_back(neuron(input_size,func)); 
}
size_t layer::split_up()
{
	static size_t number_of_threads; 
	if(threads.is_started())
		 number_of_threads = threads.start(); 
	assert(m_neurons.size() % number_of_threads == 0);
	size_t split = m_neurons.size() / number_of_threads; 
	return split; 
}

void layer::forward_layer(value_array& in)
{
	/*std::for_each(m_neurons.begin(), m_neurons.end(), [&](neuron & n){
			n.set_input(in);	
			});
	auto function = [](std::vector<neuron>& neural){
		std::for_each(neural.begin(), neural.end(),[&](neuron& n){
		n.forward(); 
		}); 
				},
	size_t split = split_up(); 
	for(size_t i = 0 ; i< m_neurons.size(); i = i+split){
		threads.enqueue( );
	}*/
}
void layer::normal_forward_layer(value_array& in)
{
	std::vector<value> temp; 
	std::for_each(m_neurons.begin(), m_neurons.end(), [&](neuron& n)  {
			n.set_input(in);
			n.forward();
			temp.push_back(); 
			}); 
	final = temp;
	return;
}
value_array& layer::layer_output()
{
	return final;
}
