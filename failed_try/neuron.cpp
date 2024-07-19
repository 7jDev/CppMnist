#include "neural.h"
#include <iostream>
threadPool<std::vector<value_array>, std::vector<value>> layer::pool; 
neuron::neuron(){}
neuron::neuron(size_t input, activations act): weights(input), neuron_activation(act) 
{
	weights.random_init();
	weights.requires_grad();
	bias.random_init();
	bias.requires_grad();
}
void neuron::n_output(value_array& input)
{
	weights_input = std::move(input* weights);  
	sum = std::move(weights_input.sum());
	sumwbias = std::move(sum + bias); 
	switch(neuron_activation){ 
		case TANH: 	
	 	out = std::move(sumwbias.tanh()); 
		break;
		case ELU:
	 	out = std::move(sumwbias.elu());
		case NONE:
	       break;	
	}
	return;
}

activations neuron::return_activation()
{
	return neuron_activation;
}

value& neuron::return_output() 
{
	if (neuron_activation== NONE)
		return sumwbias;
	if(neuron_activation!= NONE)
		return out;
	return out; 
}
std::ostream& operator<<(std::ostream& os, neuron& input){
	return os << "weights\n" << input.weights; 
}
layer::layer(){}
layer::layer(size_t amount, size_t input, activations act): m_size(amount), m_input(input), output(amount), m_func(act)
{
	next =nullptr; 
	input_normal.reserve(amount); 
	m_neurons.reserve(amount);
       	for(size_t i=0; i<amount; i++)
	{
		neuron temp(input,m_func);
		m_neurons.push_back(std::move(temp)); 
	}
}
std::vector<std::tuple<std::vector<value_array>,std::vector<std::reference_wrapper<neuron>>>> layer::split_neurons(value_array& inputs)
{
	size_t number = layer::pool.start();
	assert(m_neurons.size() % number == 0);
	std::vector<std::tuple<std::vector<value_array>,std::vector<std::reference_wrapper<neuron>>>> answer; 
	size_t x = m_neurons.size()/number; 
	for(size_t z = 0; z<number; z++){
		std::tuple<std::vector<value_array> ,std::vector<std::reference_wrapper<neuron>>> temp; 
		for(size_t t = 0; t<x; t++){
			std::get<0>(temp).push_back((inputs.return_copy()));
			std::get<1>(temp).push_back(std::ref(m_neurons[t]));
		}
		answer.push_back(std::move(temp));
	}
	return answer; 
}
std::vector<value> layer::apply(std::vector<value_array>& input,std::vector<std::reference_wrapper<neuron>>& neuro) //std::tuple<std::vector<value_array>, std::vector<std::reference>>
{
	std::vector<value> answer;
	answer.reserve(neuro.size());
	std::vector<value_array>::iterator input_it= input.begin(); 
	std::vector<std::reference_wrapper<neuron>>::iterator neuro_it= neuro.begin(); 
	while(input_it!= input.end() && neuro_it != neuro.end())
	{

	neuro_it->get().n_output(*input_it);
	answer.push_back(neuro_it->get().return_output().return_copy());
	input_it++; neuro_it++; 
	}
	assert(answer.size() == neuro.size());
	return answer; 
}
void layer::set_input(value_array& input)
{
assert(input_normal.capacity() > 0); 
for(size_t _ = 0; _<m_input; _++)
	input_normal.push_back(std::move(input.return_copy()));
}
void layer::normal_output(value_array& input)
{
	set_input(input);
	std::vector<value> temp;
	std::vector<value_array>::iterator input_it = input_normal.begin(); 
	std::vector<neuron>::iterator neuro_it= m_neurons.begin(); 
	while(input_it!= input_normal.end() && neuro_it!= m_neurons.end())
	{

	neuro_it->n_output(*input_it);
	temp.push_back(std::move(neuro_it->return_output()));
	neuro_it++; input_it++;
	}
	output = temp; 
	return;
}
void layer::l_output(value_array& input)
{
	std::vector<std::tuple<std::vector<value_array>,std::vector<std::reference_wrapper<neuron>>>> split = split_neurons(input);
	std::vector<value> temp;
	for(std::tuple<std::vector<value_array>,std::vector<std::reference_wrapper<neuron>>>& block:split)
	{
		std::function<std::vector<value>(std::vector<value_array>&)> func = std::bind(&layer::apply, this,std::placeholders::_1,(std::get<std::vector<std::reference_wrapper<neuron>>>(block)));
		std::function<std::vector<value>(std::vector<value_array>&)> real_fn= [func](std::vector<value_array>& i){return func(i);};
		layer::pool.jobQueue(real_fn,std::get<std::vector<value_array>>(block));
	}
	std::vector<std::vector<value>> out = std::move(layer::pool.return_output());
	for(std::vector<value>& subarr: out)
		temp.insert(temp.begin(), std::make_move_iterator(subarr.begin()), std::make_move_iterator(subarr.end()));
	output = temp; 
	return;
}
activations layer::return_activation()
{
	return m_neurons[0].return_activation(); 
}
void layer::set_next(layer* i)
{
	next = i;
}
layer * layer::return_this()
{
	return this; 
}
layer * layer::get_next()
{
	return next; 
}
void layer::softmax()
{
 	soft_output = std::move(output.softmax());
}
value layer::cross_entropy(int i) {
	return soft_output.cross_entropy(i);
}
value_array& layer::return_l_output()
{
if (return_activation() ==  NONE)
return soft_output;	
return output;
}
mlp::mlp(){}
mlp::mlp(int input_size ,std::initializer_list<int> neurons, std::initializer_list<activations> activation_func):
	layer_neurons(std::move(neurons)),
       	functions(std::move(activation_func)),
	current_data_input(input_size)
{
	current_path = "./Reduced/Training/";
	current_directory.reserve(1000);
	current_class = 0;
	layers.reserve(neurons.size() +2 ); 
	layers.push_back(std::move(layer(input_size, input_size, TANH)));	
	for(size_t i=1; i< neurons.size()+1; i++){
		if(i==1)
			layers.push_back(std::move(layer(layer_neurons[i-1], input_size, functions[i-1])));
		else
			layers.push_back(std::move(layer(layer_neurons[i-1], layer_neurons[i-2], functions[i-1])));
	}
	layers.push_back(std::move(layer(10,layer_neurons[layer_neurons.size()-1], NONE )));
	for(size_t i=0; i<layers.size(); i++)
	{
		layers[i].set_next(layers[i+1].return_this());
	}
}
std::vector<std::string> mlp::delimit(const std::string& input, const char stop )
{
	std::vector<std::string> answer; 
	int temp =0; 
	for(size_t i=0; i< input.size(); i++)
	{
		if(input[i] == stop)
		{
		answer.push_back(input.substr(temp, i-temp));
		temp = i+1;
		}
	}
	if(input.substr(temp, input.size() -1) != "") 
		answer.push_back(input.substr(temp, input.size() -1));
	return answer;
}
std::vector<std::string>& mlp::start(std::string& path,const char n_class)
{
	path.push_back(n_class);
	for(const auto& entry: std::filesystem::directory_iterator(path))
		current_directory.push_back(entry.path().string());
	return current_directory;
}
std::vector<double> mlp::get_file(std::string& path, const char n_class)
{
	if(current_directory.size() == 0){
	start(path, n_class);
	}
	std::string line;
	std::ifstream file(current_directory[current_directory.size()-1]);
	if(file.is_open())
	{
	getline(file,line); 
	}
	std::vector<std::string> temp= std::move(delimit(line, ' '));
	std::vector<double> values; 
	values.reserve(784);
	std::for_each(temp.begin(), temp.end(), [&](std::string& x){
			values.push_back(stod(x)/255.0);
			});
	current_directory.erase(current_directory.end()-1);
        return values;	
}
char mlp::get_current_class()
{
	if(!current_directory.size() && current_class == 9){
		current_class = -1;
		return 0;
	}
	if(!current_directory.size()){ 
		current_class += 1; 
		current_path.erase(current_path.end()-1);
	}
       	return current_class + '0';	
}
value_array& mlp::get_data(std::string& path)
{
	static bool init = false;
	std::vector<double> temp;
	if (!init){
		temp =  std::move(get_file(path,'0'));
	init = true;
	}
	else 
	temp =  std::move(get_file(path,get_current_class()));	
	current_data_input = value_array(temp);
	return current_data_input;
}

value_array& mlp::predict_helper(layer& current ,value_array& input)
{
	if(current.return_activation() == NONE){
		current.normal_output(input);
		current.softmax(); 
		loss = std::move(current.cross_entropy(current_class));	
		return current.return_l_output();
	}
	current.normal_output(input); 
	return predict_helper(*current.get_next(), current.return_l_output()); 
}
value_array& mlp::predict(value_array& input){
	return predict_helper(layers[0], input);
	
}
void mlp::one_epoch_h(value_array & input){
	value_array& real_input = input; 
	using namespace std;
	while(current_class !=9 && current_directory.size() != 0){
	static int ex = 0;
	ex += 1;
	std::cout << "on number: " << ex << "	loss: " << loss << std::endl;
	predict(real_input);
	calculate_gradients();
	loss.learn(0.01);
	real_input = std::move(get_data(current_path));
	}
	return;
}
value& mlp::one_epoch()
{
one_epoch_h(get_data(current_path));
return loss;
} 
void mlp::calculate_gradients()
{
	loss.calculate_gradients();
}

