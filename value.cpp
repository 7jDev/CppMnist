#include "value.h"
value::value(double data): m_data(data), m_gradient(0),m_node{this, nullptr}{}
value::value(value&& other): m_data(other.m_data),
       	m_gradient(other.m_gradient),
       	m_parents(std::move(other.m_parents))
{
	other.m_data = 0;
	other.m_gradient = 0; 
}
value value::operator+(value& other){
	value answer(m_data + other.m_data);
	m_gradient = 1;
	other.m_gradient = 1; 
	answer.m_parents.push_back(this);
	answer.m_parents.push_back(other.m_ptr);
	return answer;
};
value value::operator*(value& other){
	value answer(m_data * other.m_data);
	m_gradient = other.m_data;
	other.m_gradient = m_data; 
	answer.m_parents.push_back(this);
       answer.m_parents.push_back(other.m_ptr);
	return answer;       
}
void value::random_init(){
	std::random_device rd;
	std::uniform_real_distribution<double> dist(-1,1);
	m_data = dist(rd);
}
value value::tanh(){
value answer((exp(m_data)-exp(-(m_data)))/(exp(m_data)+ exp(-(m_data))));
m_gradient = 1 - pow(answer.m_data,2);
answer.m_parents.push_back(this);
return answer; 
}
void value::calculate_gradients(){
m_gradient =1;
queue<value*> Queue;
Queue.enqueue(m_node);
while(Queue.peek() != nullptr){
	struct Node<value*> current = *Queue.peek();
	std::for_each(Queue.peek()->data->m_parents.begin(), Queue.peek()->data->m_parents.end(), [&](value* temp){
			Queue.enqueue(temp->m_node);
			temp->m_gradient *= current.data->m_gradient;
			});
	Queue.dequeue();
}
return;
}
std::ostream& operator<<(std::ostream& os, value & input){
	return os << "data: " << input.m_data << std::endl << "gradient: " << input.m_gradient << std::endl;
}
