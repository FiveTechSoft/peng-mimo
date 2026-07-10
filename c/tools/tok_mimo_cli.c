/* CLI di validazione: legge tutto stdin (UTF-8, puo' contenere \n/\t), lo
 * codifica con tok.h (senza token speciali) e stampa gli id come array JSON.
 * uso:   ./tok_mimo_cli <dir-modello>  < testo
 *        (si aspetta <dir-modello>/tokenizer.json)
 * build: gcc -O2 -o tok_mimo_cli tools/tok_mimo_cli.c -lm */
#define _GNU_SOURCE
#include "../tok.h"

int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"uso: %s <model_dir> < testo\n", argv[0]); return 1; }
    char path[4096];
    snprintf(path,sizeof(path),"%s/tokenizer.json",argv[1]);

    Tok T;
    tok_load(&T, path);
    fprintf(stderr,"caricato: vocab_ids=%d specials=%d\n", T.n_ids, T.nsp);

    size_t cap=1<<20, len=0;
    char *buf=malloc(cap), *nbuf; size_t nr;
    while((nr=fread(buf+len,1,cap-len,stdin))>0){
        len+=nr;
        if(len==cap){ cap*=2; nbuf=realloc(buf,cap); if(!nbuf){fprintf(stderr,"oom\n");return 1;} buf=nbuf; }
    }

    int max=(int)len+16;
    int *ids=malloc(sizeof(int)*(size_t)max);
    int n=tok_encode(&T, buf, (int)len, ids, max);

    printf("[");
    for(int i=0;i<n;i++) printf(i ? ",%d" : "%d", ids[i]);
    printf("]\n");
    return 0;
}
