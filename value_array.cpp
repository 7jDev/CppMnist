#include "value.h"
value_array::value_array(){}
value_array::value_array(size_t length): len(length){
	m_values.reserve(length);
	for(size_t i=0; i< length; i++){
	value temp;
	m_values.push_back(std::move(temp));
	}
	
}
value& value_array::operator[](const size_t i){
	return m_values[i];
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
	len = temp.size();
	return *this;
}
value_array value_array::return_copy()
{
	value_array answer(len);
	for(size_t i=0; i<len; i++)
	{
		 answer.m_values.push_back(std::move(m_values[i].return_copy()));
	}
	return answer;	
}
value_array value_array::operator+(value_array& other){
	assert(other.len == len); 
	value_array answer(len); 
	for(size_t i=0; i<len; i++){
		 answer[i] = std::move (m_values[i] + other[i]);
	}
	return answer;
}
value_array value_array::operator*(value_array& other){
	assert(other.len == len); 
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
	value_array answer(len);
	double x[len];
       	double sum;
	for(auto& val: m_values)
		sum += exp(val.return_data()); 
	for(size_t i =0; i< len; i++)
		answer.m_values[i] = exp(m_values[i].return_data())/sum; 
     	std::for_each(answer.m_values.begin(), answer.m_values.end(), [&](value& x){
			x.change_gradient((x.return_data() * (1 - x.return_data()) - x.return_data() - x.return_data()));  	
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
