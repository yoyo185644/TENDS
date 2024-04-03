''' 自定义分页组件
 以后如果想使用这个分页组件，需要做如下几件事
 在视图函数中：
     def pretty_list(request):
        # 根据自己的情况去筛选自己的数据
        queryset = MyModel.PrettyNum.objects.filter(**data_dict).order_by('-level')
        # 实例化分页对象
        page_object = Pagination(request,queryset)

        context = {
            "search_data":search_data,
            "queryset":page_object.page_queryset,    # 分完页的数据
            "page_string":page_object.html()         # 生成页吗
        }
        return render(request,'pretty_list.html',context)
在html页面中：
    <div class="clearfix">
        <ul class="pagination" >
            {{ page_string }}
        </ul>
    </div>

 '''
from django.utils.safestring import mark_safe

class Pagination(object):
    def __init__(self, request, queryset, page_size=6, page_param='page', plus=5):
        page = request.GET.get(page_param, '1')
        if page.isdecimal():
            page = int(page)
        else:
            page = 1
        self.page = page
        self.page_size = page_size
        self.start = (page - 1) * page_size
        self.end = page * page_size
        self.page_queryset = queryset[self.start:self.end]

        total_count = queryset.count()
        total_page_count, div = divmod(total_count, self.page_size)
        if div:
            total_page_count += 1
        self.total_page_count = total_page_count
        self.plus = plus

    def html(self):
        page_str_list = []
        page_str_list.append("<li><a data-page='{}'>First</a></li>".format(1))

        # Previous page
        if self.page > 1:
            prev = "<li><a data-page='{}'>Prev</a></li>".format(self.page - 1)
        else:
            prev = "<li><a data-page='{}'>Prev</a></li>".format(1)
        page_str_list.append(prev)

        # Current page
        page_str_list.append("<li class='active'><a data-page='{}'>{}</a></li>".format(self.page, self.page))

        # Next page
        if self.page < self.total_page_count:
            next = "<li><a data-page='{}'>Next</a></li>".format(self.page + 1)
        else:
            next = "<li><a data-page='{}'>Next</a></li>".format(1)
        page_str_list.append(next)

        # Last page
        page_str_list.append("<li><a data-page='{}'>Last</a></li>".format(self.total_page_count))

        search_string = '''
        <li>
        <form style='float:left;margin-left:-1px' method='get'>
        <input name='page' value=''
        style='position:relative;float:left;display:inline-block;width:80px;border-radius:0;'
        type="text" class="form-control" placeholder="Page number">
        <button style='border-radius:0' class='btn btn-default' type='submit'>Go</button>
        </form>
        </li>
        '''
        page_str_list.append(search_string)
        page_string = mark_safe("".join(page_str_list))
        return page_string




