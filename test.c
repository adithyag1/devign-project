void test(char *str) {
    char buffer[10];
    strcpy(buffer, str); // Vulnerable: No bounds check
}