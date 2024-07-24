#include "value.h"
value_array::value_array(){}
value_array::value_array(size_t length): len(length){
	m_values.reserve(length);
	for(size_t i=0; i< length; i++){
	value temp;
	m_values.push_back(std::move(temp));
	}
}
inline std::vector<value>& value_array::return_vector()
{
	return m_values;
}
/*void value_array::fill_with(size_t i)
{
	std::for_each(m_values.begin(), m_values.end(),[&](value& x){
			x = std::move(value(i));
			});
}*/

double value_array::clip(double input, double lower, double higher){
	return std::max(lower, std::min( input,higher));
} 
value& value_array::operator[](const size_t i){
	return m_values[i];
}
value_array::value_array(std::vector<double>& items): len(items.size())
{

	m_values.reserve(items.size());
	std::for_each(items.begin(), items.end(),[&](double& item){
			m_values.push_back(value(item));
			});
}
value_array::value_array(value_array&& other): len(other.len), m_values(std::move(other.m_values)) {}
void value_array::random_init(){
	std::for_each(m_values.begin(), m_values.end(),[](value& item){
		item.random_init();			
	});
	return; 
}
value_array& value_array::operator=(std::vector<value>& temp)
{
	m_values = std::move(temp);
	len = m_values.size();
	return *this;
}
size_t value_array::size(){
return m_values.size(); 
}
value_array value_array::return_copy()
{
	value_array answer(m_values.size());
	for(size_t i=0; i<m_values.size(); i++)
	{
		 answer.m_values[i] = std::move(m_values[i].return_copy());
	}
	
	return answer;	
}
value_array value_array::operator+(value_array& other){
	assert(other.m_values.size()== m_values.size()); 
	value_array answer(len); 
	for(size_t i=0; i<len; i++){
		 answer[i] = std::move (m_values[i] + other[i]);
	}
	return answer;
}
value_array value_array::operator*(value_array& other){
	assert(other.m_values.size()== m_values.size());
	value_array answer(len); 
	for(size_t i=0; i<len; i++){
		 answer[i] = std::move (m_values[i] * other[i]);
	}
	return answer;
}
value_array& value_array::operator=(value_array&& other)
{
	m_values = std::move(other.m_values);
	other.m_values.clear();
	len = other.len;
	other.len =0;
	return *this;
}
value value_array::sum(){
	value answer;
	std::for_each(m_values.begin(), m_values.end(), [&](value& input){
			answer += input; 	
			});
       return answer; 	
}
value_array value_array::softmax()
{
	value_array answer(m_values.size());
	double x[m_values.size()]; 
       	double sum{};
	for(auto& val: m_values)
		sum += exp(val.return_data()); 
	for(size_t i =0; i< m_values.size(); i++){
		answer.m_values[i] = exp(m_values[i].return_data())/sum; 
		answer.m_values[i].push_back(m_values[i]);
	}
       	std::for_each(m_values.begin(), m_values.end(), [&](value& z){
			z.change_gradient(1);
			});
	return answer; 	
}
value value_array::cross_entropy(int i)
{
	double val = m_values[i].return_data() + 1e-8; 
	value answer(-log( clip(val, 0.0, 1.0)));
	std::for_each(m_values.begin(), m_values.end(), [&](value& x){
			answer.push_back(x);
			if(m_values[i]==x)
				x.change_gradient(x.return_data() -1); 
			else
				x.change_gradient(x.return_data());
			});
	return answer;
}
void value_array::requires_grad()
{
	for(auto& item: m_values){
		item.requires_grad();
	}
}
std::ostream& operator<<(std::ostream& os, value_array& input){
	std::for_each(input.m_values.begin(), input.m_values.end(), [&](value& x){
			os << x; 
			});
	return os;
}
