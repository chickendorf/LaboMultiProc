#include <stdlib.h>
#include <limits.h>

// Linked list struct
typedef struct node {
  int val;
  struct node *next;
  int to_remove;
} node_t;

typedef struct {
  node_t *head;
  omp_lock_t *lock;
} linked_list;

/* This function intializes a thread-safe linked-list working with a single lock that
*controls the access to its entire data structure.
*/
linked_list* init_linked_list(){
  node_t *head = malloc(sizeof(node_t));
  head->val = INT_MIN;
  head->next = NULL;
  head->to_remove = 0;

  linked_list *list = malloc(sizeof(linked_list));
  list-> head = head;
  omp_lock_t lock;
  list->lock = &lock;
  omp_init_lock(list->lock);

  return list;
}

/* This function inserts a new given element at the right position of a given linked list.
* It returns 0 on a successful insertion, and -1 if the list already has that value.
*/
int insert(linked_list *list, int val) {
  omp_set_lock(list->lock);
  node_t *head = list->head;
  node_t *previous, *current;
  current = head;

  while (current && current->val < val) {
    previous = current;
    current  = current->next;
  }

  if (current && current->val == val) { // This value already exists!
    omp_unset_lock(list->lock);
    return -1;
  }

  // Here is the right position to insert the new node.
  node_t *new_node;
  new_node = malloc(sizeof(node_t));
  new_node->val = val;
  new_node->next = current;
  new_node->to_remove = 0;

  previous->next = new_node;

  omp_unset_lock(list->lock);
  return 0;
}

/* This function removes the specified element of a given linked list.
* The value of that element is returned if the element is found; otherwise it returns -1.
*/
int delete(linked_list *list, int val) {
  omp_set_lock(list->lock);
  node_t *head = list->head;
  node_t *previous, *current;

  if (head->next == NULL) { // The list is empty.
    omp_unset_lock(list->lock);
    return -1;
  }

  previous = head;
  current = head->next;
  while (current) {
    if (current->val == val) {
      previous->next = current->next;
      current->to_remove = 1; // Another system component will free this node later
      return val;
    }

    previous = current;
    current  = current->next;
  }

  omp_unset_lock(list->lock);
  return -1;

}


/* This function searches for a specified element in a given linked list.
* It returns zero if the element is found; otherwise it returns -1.
*/
int search(linked_list *list, int val) {
  omp_set_lock(list->lock);
  node_t *head = list->head;
  node_t *current = head->next;

  while (current) {
    if (current->val == val) {
      omp_unset_lock(list->lock);
      return 0;
    }
    current  = current->next;
  }


  omp_unset_lock(list->lock);
  return -1;

}
