#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "linked-list-2.c"

void print_list(linked_list *list){

  node_t *head = list->head;

  printf("%d->", head->val);
  node_t *current = head->next;

  while (current) {
    printf("%d->", current->val);
    current  = current->next;
  }
}

int main (int argc, const char *argv[]) {

  int threads;

  if (argc != 2) {
    printf("Invalid input! \nUsage: ./assignment3_2 <threads>\n");
    return 1;
  } else {
    threads = atoi(argv[1]);
  }

  omp_set_num_threads(threads);
  linked_list *list = init_linked_list();

  #pragma omp for schedule(static)
  for(int i = 100; i > 0; i--){
    insert(list, i);
  }
  print_list(list);
}
