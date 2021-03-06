#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "linked-list.c"

void print_list(node_t *head){

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
    printf("Invalid input! \nUsage: ./assignment3 <threads>\n");
    return 1;
  } else {
    threads = atoi(argv[1]);
  }

  omp_set_num_threads(threads);

  node_t *head = malloc(sizeof(node_t));
  head->val = INT_MIN;
  head->to_remove = 0;
  head->next = NULL;

  #pragma omp for schedule(static)
  for(int i = 100; i > 0; i--){
    insert(head, i);
  }
  print_list(head);
}
