#include <iostream>
#ifndef queue_H
#define queue_H
template <typename T> struct Node{
	T data;
	struct Node<T>* next;
};
template <typename T> class queue{
	private: 
		size_t length;
		struct Node<T>* head;
		struct Node<T>* tail;
	public:
	queue(){
        head = nullptr;
        tail =nullptr;
        length = 0;
	}	
	struct Node<T>* peek(){
                if(head == nullptr){
                        return nullptr;
                }
                return head;
        }
	void enqueue(struct Node<T>& input){
                if(length == 0){
                head = &input;
                tail = &input;
                length++;
                return;
                }
                length++;
                struct Node<T>& temp = *(this->tail);
                this->tail = &input;
                temp.next = tail;       
                return;
        }

	void dequeue(){
        try{
        if(length==0){
                throw std::runtime_error("length is zero");
        }
        if (length == 1){
                head = nullptr;
                tail = nullptr;
                length--;
                return;
        }
        length--;
        struct Node<T> temp = *head;
        head = temp.next;
        temp.next = nullptr;
        }catch(const std::exception& error){
                std::cout << error.what() <<std::endl;
        }
        return; 
        }

}; 
#endif
