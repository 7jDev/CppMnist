#include "value.h"
value_array::value_array(size_t length): len(length){
	m_values.reserve(length);
	for(size_t i=0; i< length; i++){
	value temp;
	m_values.push_back(std::move(temp));
	}
	
}
value& operator[](size_t i){
	return m_values[i];
}
value& operator[](size_t i) const{
	return m_values[i];
}
value_array::value_array(value_array&& other): len(other.len), m_values(std::move(other.m_values)) {}
void value_array::random_init(){
	std::for_each(m_values.begin(), m_values.end(),[](value& item){
		item.random_init();			
	});
	return; 
}
value_array value_array::operator+(value_array& other){
	assert(other.len == len); 
	value_array answer(len); 
	for(size_t i=0; i<len; i++){
		 answer.m_values[i] = std::move (m_values[i] + other.m_values[i]);
	}
	return answer;
}
value_array value_array::operator*(value_array& other){
	assert(other.len == len); 
	value_array answer(len); 
	for(size_t i=0; i<len; i++){
		 answer.m_values[i] = std::move (m_values[i] * other.m_values[i]);
	}
	return answer;
}
value value_array::sum(){
	value answer;
	std::for_each(m_values.begin(), m_values.end(), [&](value& input){
			answer += input; 	
			});
       return answer; 	
}
value value_array::softmax()
std::ostream& operator<<(std::ostream& os, value_array& input){
	std::for_each(input.m_values.begin(), input.m_values.end(), [&](value& x){
			os << x; 
			});
	return os;
}
