#include "value.h"
value::value(): m_gradient(0),m_data(0){}
value::value(double data): m_data(data), m_gradient(0){}
value::value(value&& other): m_data(other.m_data),
       	m_gradient(other.m_gradient),
       	m_parents(std::move(other.m_parents)),
	m_grad(other.m_grad)
{
	other.m_data = 0;
	other.m_gradient = 0; 
}
value value::operator+(value& other){
	value answer(m_data + other.m_data);
	m_gradient = 1;
	other.m_gradient = 1; 
	answer.m_parents.push_back(std::ref(*this));
	answer.m_parents.push_back(std::ref(other));
	return answer;
};
value value::operator*(value& other){
	value answer(m_data * other.m_data);
	m_gradient = other.m_data;
	other.m_gradient = m_data;
	answer.m_parents.push_back(std::ref(*this));
	answer.m_parents.push_back(std::ref(other));
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
bool value::operator==(value& other)
{
return m_data == other.m_data;
}
value value::return_copy()
{
	value answer; 
	answer.m_data = m_data; 
	answer.m_gradient = m_gradient;
	answer.m_parents.push_back(std::ref(*this));
	m_gradient = 1; 
	return answer;	
}
void value::push_back(value & item)
{
	m_parents.push_back(std::ref(item));
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
	m_parents.push_back(std::ref(other));
	return *this; 
}
value value::tanh(){
value answer((exp(m_data)-exp(-(m_data)))/(exp(m_data)+ exp(-(m_data))));
m_gradient = 1 - pow(answer.m_data,2);
answer.m_parents.push_back(std::ref(*this));
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
answer.m_parents.push_back(std::ref(*this));
return answer;
}

void value::calculate_gradients(){
m_gradient =1;
std::queue<std::reference_wrapper<value>> Queue;
Queue.push(std::ref(*this));
while(Queue.size() !=0){
	while(Queue.front().get().m_parents.empty()){
		Queue.pop();
		if(Queue.size() ==0 )
			break;
	}
	if(Queue.size() == 0)
		break; 
	std::reference_wrapper<value>& current = Queue.front();	
	std::for_each(current.get().m_parents.begin(), current.get().m_parents.end(), [&](value& temp){
			Queue.emplace(std::ref(temp));
			temp.m_gradient *= current.get().m_gradient;
			});
	Queue.pop();
}
return;
}
void value::learn(double lr){
std::queue<std::reference_wrapper<value>> Queue;
Queue.emplace(std::ref(*this));
while(!Queue.empty()){
	std::reference_wrapper<value> current = Queue.front();
	std::for_each(current.get().m_parents.begin(), current.get().m_parents.end(), [&](value& temp){
			Queue.emplace(std::ref(temp));
			if(temp.m_grad == true){
			temp.m_data = temp.m_data - (current.get().m_gradient * lr);
			}
			});
	Queue.pop();
}
return;
}
std::ostream& operator<<(std::ostream& os, value & input){
	return os << "data: " << input.m_data << std::endl << "gradient: " << input.m_gradient << std::endl;
}
