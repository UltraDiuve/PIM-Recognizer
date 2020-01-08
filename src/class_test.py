class A(object):
    def foo(self):
        print('from top : foo')

    def baz(self):
        print('from top : baz')
        self.foo()


class B(A):
    def baz(self):
        print('from bottom : baz')
        super().baz()

    def foo(self):
        print('from bottom : foo')
        super().foo()
