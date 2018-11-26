// Linked list struct
typedef struct node {
  int val;
  struct node *next;
  int to_remove;
  omp_lock_t *lock;
} node_t;

/* This function inserts a new given element at the right position of a given linked list.
* It returns 0 on a successful insertion, and -1 if the list already has that value.
*/
int insert(node_t *head, int val) {
  node_t *previous, *current;
  current = head;

  while (current && current->val < val) {
    previous = current;
    omp_set_lock(previous->lock);
      //printf("locked %d\n", previous->val);
    current  = current->next;
    omp_unset_lock(previous->lock);
    //    printf("unlocked %d\n", previous->val);
  }

  if (current && current->val == val) { // This value already exists!
    return -1;
  }

  // Here is the right position to insert the new node.
  node_t *new_node;
  new_node = malloc(sizeof(node_t));
  new_node->val = val;
  new_node->next = current;
  new_node->to_remove = 0;
  omp_lock_t lock;
  new_node->lock = &lock;
  omp_init_lock(new_node->lock);
  //printf("init %d\n", new_node->val);
  previous->next = new_node;
  return 0;
}

/* This function removes the specified element of a given linked list.
* The value of that element is returned if the element is found; otherwise it returns -1.
*/
int delete(node_t *head, int val) {
  node_t *previous, *current;

  if (head->next == NULL) { // The list is empty.
    return -1;
  }
  previous = head;
  current = head->next;
  while (current) {
    omp_set_lock(current->lock);
    if (current->val == val) {
      previous->next = current->next;
      current->to_remove = 1; // Another system component will free this node later
      omp_unset_lock(current->lock);
      omp_destroy_lock(current->lock);
      return val;
    }

    previous = current;
    current  = current->next;
    omp_unset_lock(previous->lock);
  }

  return -1;
}


/* This function searches for a specified element in a given linked list.
* It returns zero if the element is found; otherwise it returns -1.
*/
int search(node_t *head, int val) {
  node_t *current = head->next;

  while (current) {
    omp_set_lock(current->lock);
    if (current->val == val) {
      omp_unset_lock(current->lock);
      return 0;
    }
    node_t *tmp = current;
    current  = current->next;
    omp_unset_lock(tmp->lock);
  }

  return -1;
}
