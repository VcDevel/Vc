#include <Vc/simdize>
#include <Vc/IO>

template <typename float_t> struct PointTemplate
{
    float_t x, y, z;

    template <std::size_t I> friend inline float_t &get(PointTemplate<float_t> &p)
    {
        static_assert(I <= 2, "get<N>(PointTemplate<T>) requires N <= 2");
        if (I == 0) {
            return p.x;
        } else if (I == 1) {
            return p.y;
        } else if (I == 2) {
            return p.z;
        }
    }
    template <std::size_t I>
    friend inline const float_t &get(const PointTemplate<float_t> &p)
    {
        static_assert(I <= 2, "get<N>(PointTemplate<T>) requires N <= 2");
        if (I == 0) {
            return p.x;
        } else if (I == 1) {
            return p.y;
        } else if (I == 2) {
            return p.z;
        }
    }

    static constexpr std::size_t tuple_size = 3;
};

using Point = PointTemplate<float>;

template <typename T> class Vectorizer
{
public:
    using value_type = Vc::simdize<T>;

    void append(const T &x) { decorate(data)[count++] = x; }
    bool isFull() const { return count == data.size(); }
    const value_type &contents() const { return data; }

private:
    value_type data;
    int count = 0;
};

int main()
{
    Vectorizer<Point> v;
    float x = 0.f;
    while (!v.isFull()) {
        v.append(Point{x, x + 1, x + 2});
        x += 10;
    }
    const auto &vec = v.contents();
    std::cout << vec.x << '\n' << vec.y << '\n' << vec.z << '\n';
    return 0;
}
