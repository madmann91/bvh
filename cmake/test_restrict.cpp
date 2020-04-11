int set(int* restrict a, const int* restrict b) {
    *a = *b;
}

int main(int argc, char* argv) {
    int a, b = 0;
    set(&a, &b);
    return a;
}
