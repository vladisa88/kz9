import Vue from 'vue'
import './plugins/axios'
import App from './App.vue'
import VueCarousel from 'vue-carousel';

import 'bulma/css/bulma.min.css';
import 'bulma-switch/dist/css/bulma-switch.min.css'
import 'vue2-dropzone/dist/vue2Dropzone.min.css';

import VueImageCompare from 'vue-image-compare';

Vue.use(VueImageCompare);
Vue.use(VueCarousel);

Vue.config.productionTip = false

new Vue({
  render: h => h(App),
}).$mount('#app')
