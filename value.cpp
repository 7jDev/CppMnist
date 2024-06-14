#include "value.h"
value::value(): m_gradient(0),m_data(0),m_node{this, nullptr} {}
value::value(double data): m_data(data), m_gradient(0),m_node{this, nullptr}{}
value::value(value&& other): m_data(other.m_data),
       	m_gradient(other.m_gradient),
       	m_parents(std::move(other.m_parents)),
	m_grad(other.m_grad)
{
	m_node.data = this; 
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
value& value::operator=(value&& other)
{
	m_data = other.m_data;
	m_gradient = other.m_gradient;
	m_parents = std::move(other.m_parents);
	other.m_data = 0;
	other.m_gradient = 0; 
	return *this;
}
value value::return_copy()
{
	value answer; 
	answer.m_data = m_data; 
	answer.m_gradient = m_gradient;
	answer.m_parents.push_back(m_ptr);
	m_gradient = 1; 
	return answer;	
}
void value::push_back(value * item)
{
	m_parents.push_back(item);
}
void value::change_gradient(double x)
{
	m_gradient = x; 
}
double value::return_data()
{	
	return m_data;
}
void value::random_init(){
	std::random_device rd;
	std::normal_distribution<double> dist;
	m_data = dist(rd);
}
void value::requires_grad(){
	m_grad = true;
}
value& value::operator+=(value& other){
	m_data += other.m_data;
       	other.m_gradient = 1; 	
	m_parents.push_back(other.m_ptr);
	return *this; 
}
value value::tanh(){
value answer((exp(m_data)-exp(-(m_data)))/(exp(m_data)+ exp(-(m_data))));
m_gradient = 1 - pow(answer.m_data,2);
answer.m_parents.push_back(this);
return answer; 
}
value value::elu(){
value answer;
double alpha= {0.1};
if (m_data<0){
        answer.m_data =(alpha *exp(m_data) -1 ); 
        m_gradient = alpha * exp(m_data); 
}else{
        answer.m_data = m_data; 
        m_gradient = 1; 
}
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
void value::learn(double lr){
queue<value*> Queue;
Queue.enqueue(m_node);
while(Queue.peek() != nullptr){
	struct Node<value*> current = *Queue.peek();
	std::for_each(Queue.peek()->data->m_parents.begin(), Queue.peek()->data->m_parents.end(), [&](value* temp){
			Queue.enqueue(temp->m_node);
			if(temp->m_grad == true){
			temp->m_data = temp->m_data - (current.data->m_gradient * lr);
			}
			});
	Queue.dequeue();
}
return;
}
std::ostream& operator<<(std::ostream& os, value & input){
	return os << "data: " << input.m_data << std::endl << "gradient: " << input.m_gradient << std::endl;
}
