<template>
  <div>
    <nav
      class="navbar is-fixed-top is-success is-transparent my-container"
      :class="{ 'bottom-border': showNavbarShadow }"
    >
      <div class="navbar-brand">
        <a class="logo navbar-item" href="#main-section">
          <svg version="1.0" xmlns="http://www.w3.org/2000/svg"
        width="1200.000000pt" height="1200.000000pt" viewBox="0 0 1200.000000 1200.000000"
        preserveAspectRatio="xMidYMid meet">

        <g transform="translate(0.000000,1200.000000) scale(0.100000,-0.100000)"
        fill="#000000" stroke="none">
        <path d="M3000 6455 l0 -2995 3000 0 3000 0 0 2995 0 2995 -3000 0 -3000 0 0
        -2995z m2580 1220 l0 -725 420 0 420 0 0 725 0 725 515 0 515 0 0 -1950 0
        -1950 -515 0 -515 0 0 775 0 775 -420 0 -420 0 0 -775 0 -775 -515 0 -515 0 0
        1950 0 1950 515 0 515 0 0 -725z"/>
        </g>
        </svg>
        </a>
      </div>
      <div class="navbar-menu navbar-end">
        <div id="navbar-content" class="navbar-item">
          <div class="field is-grouped">
            <p class="control">
              <a class="button is-ghost dark-link" href="#about-nn-section">
                <span>
                  О нейросети
                </span>
              </a>
            </p>
            <p class="control">
              <a class="button is-ghost dark-link" href="#reviews">
                <span>
                  Отзывы
                </span>
              </a>
            </p>
            <p class="control">
              <a class="button is-ghost dark-link" href="#about-us-section">
                <span>
                  О нас
                </span>
              </a>
            </p>
            <p class="control">
              <a class="button is-ghost dark-link" href="#feedback-section">
                <span>
                  Связаться с нами
                </span>
              </a>
            </p>
          </div>
        </div>
      </div>
    </nav>
    <div class="fullheight">
      <div class="main my-container" @scroll="onScroll">
        <section id="main-section" class="level columns">
          <div class="column is-full is-full-touch has-text-centered-touch main-back">
            <h1 class="main-title is-size-2-mobile">
              Анализ лояльности аудитории социальных сетей при помощи нейросети
            </h1>
            <carousel :perPageCustom="[[480, 1], [768, 1]]" :autoplay="true">
              <slide>
                <img src="./assets/images/background.jpeg">
              </slide>
              <slide>
                <img src="./assets/images/background2.png">
              </slide>
            </carousel>
            <button class="start-button button is-large is-fullwidth" @click="start">
              <span class="text">Проанализировать свое сообщество</span>
            </button>
          </div>
          <div
            class="column is-full is-flex is-flex-direction-row is-align-items-center is-hidden-touch is-hidden-desktop-only is-hidden-widescreen-only"
          >
            <!-- <figure class="image bird-image">
              <img src="./assets/images/bird.png" />
            </figure> -->
          </div>
        </section>
        <section id="about-nn-section" class="section">
          <h1 class="section-title has-text-centered mb-3">
            Как работает наша нейросеть?
          </h1>
          <div class="columns">
            <div class="column is-half">
              <div class="content has-text-justified">
                <h1>Нейросеть BERT</h1>
                <p>BERT — это нейронная сеть от Google, показавшая с большим отрывом state-of-the-art 
                  результаты на целом ряде задач. С помощью BERT можно создавать программы с ИИ 
                  для обработки естественного языка: отвечать на вопросы, заданные в произвольной форме, 
                  создавать чат-ботов, автоматические переводчики, анализировать текст и так далее.
                </p>
                <h3>Почему она?</h3>
                <p>Данная нейросеть обладает несколькими значительными преимуществами:</p>
                <ul>
                  <li>Нейронная сеть от Google.</li>
                  <li>Специализируется на анализе текстов.</li>
                  <li>Является мультиязычной.</li>
                  <li>Предобучена на огромном количестве текстов.</li>
                  <li>Дообучена специально под наш сервис.</li>
                </ul>
                <blockquote>Изначально BERT обучался на 104 языках Википедии (Многоязычный). В дополнение к многоязычной версии Google выпустила BERT на основе английской Википедии и BERT на китайском языке.</blockquote>
                <p>Наша специально обученная нейросеть позволит Вам значительно и в кратчайшие сроки улучшить качество производимого Вами контента. </p>
              </div>
            </div>
            <div class="column is-half">
              <img src="./assets/images/nn.jpg">
            </div>
          </div>
          <!-- <h2 class="subtitle pl-3 has-text-justified">
            От других нейросетей, работающих с изображениями, наша отличается
            тем, что для неё важно расположение пикселей. У раскрашивающих
            нейросетей размер изображения или соотношение сторон остаётся
            неизменным. А у сетей других типов изображение искажается по мере
            приближения к окончательной версии.
          </h2>
          <h2 class="subtitle pl-3 has-text-justified">
            Слой пулинга с функцией максимума, применяемый в классифицирующих
            сетях, увеличивает плотность информации, но при этом искажает
            картинку. Он оценивает только информацию, а не макет изображения. А
            в раскрашивающих сетях для уменьшения ширины и высоты вдвое мы
            используем шаг 2 (stride of 2). Плотность информации тоже
            увеличивается, но картинка не искажается.
          </h2>
          <h2 class="subtitle pl-3 has-text-justified">
            Также наша нейросеть отличается от прочих слоями повышения
            дискретизации (upsampling) и сохранением соотношения сторон
            изображения. Классифицирующие сети заботятся только об итоговой
            классификации, поэтому постепенно уменьшают размер и качество
            картинки по мере её прогона через нейросеть.
          </h2> -->
        </section>
        <section id="reviews" class="section">
          <h1 class="section-title has-text-centered mb-3">
            Отзывы о работе нашего сервиса
          </h1>
          <carousel :perPageCustom="[[480, 2], [768, 2]]">
            <slide>
              <div class="card">
                <div class="card-content">
                  <p class="title has-text-justified">
                    “Данный сервис дал мне невероятную мотивацию для улучшения своего контента.”
                  </p>
                  <p class="subtitle">
                    Влад А4
                  </p>
                </div>
                <footer class="card-footer">
                  <p class="card-footer-item">
                    <span>
                      Подробнее в <a href="https://twitter.com/codinghorror/status/506010907021828096">Твиттер</a>
                    </span>
                  </p>
                  <p class="card-footer-item">
                    <span>
                      Группа <a href="#"> ВКонтакте</a>
                    </span>
                  </p>
                </footer>
              </div>
            </slide>
            <slide>
              <div class="card">
                <div class="card-content">
                    <p class="title has-text-justified">
                      “Количество моих подписчиков выросло на 40% после использования сервиса”
                    </p>
                    <p class="subtitle">
                      Даня Милохин
                    </p>
                  </div>
                  <footer class="card-footer">
                    <p class="card-footer-item">
                      <span>
                        Подробнее в <a href="https://twitter.com/codinghorror/status/506010907021828096">Твиттер</a>
                      </span>
                    </p>
                    <p class="card-footer-item">
                      <span>
                        Группа <a href="#"> ВКонтакте</a>
                      </span>
                    </p>
                  </footer>
              </div>
            </slide>
            <slide>
              <div class="card">
                <div class="card-content">
                  <p class="title has-text-justified">
                    “Нейросеть неожиданно точно определяет настроение людей в комментариях.”
                  </p>
                  <p class="subtitle">
                    Артем Кацевалов
                  </p>
                </div>
                <footer class="card-footer">
                  <p class="card-footer-item">
                    <span>
                      Подробнее в<a href="https://twitter.com/codinghorror/status/506010907021828096">Твиттер</a>
                    </span>
                  </p>
                  <p class="card-footer-item">
                    <span>
                      Группа<a href="#"> ВКонтакте</a>
                    </span>
                  </p>
                </footer>
              </div>
            </slide>
            <slide>
              <div class="card">
                <div class="card-content">
                    <p class="title has-text-justified">
                      “Вау! Для меня оказалось полным шоком точность работы этого сервиса!”
                    </p>
                    <p class="subtitle">
                      Артем Амуров
                    </p>
                </div>
                <footer class="card-footer">
                  <p class="card-footer-item">
                    <span>
                      Подробнее в<a href="https://twitter.com/codinghorror/status/506010907021828096">Твиттер</a>
                    </span>
                  </p>
                  <p class="card-footer-item">
                    <span>
                      Группа <a href="#"> ВКонтакте</a>
                    </span>
                  </p>
                </footer>
              </div>
            </slide>
          </carousel>
        </section>
        <section id="about-us-section" class="section">
          <h1 class="section-title has-text-centered mb-3">
            Наша команда
          </h1>
          <div class="columns">
            <div class="column is-half">
              <div class="card">
                <div class="card-image">
                  <figure class="image is-4by3">
                    <img src="./assets/images/isakov.jpg">
                  </figure>
                </div>
                <div class="card-content">
                  <div class="media">
                    <div class="media-left">
                      <figure class="image is-48x48">
                        <img src="./assets/images/icon.jpg" alt="Placeholder image">
                      </figure>
                    </div>
                    <div class="media-content">
                      <p class="title is-4" style="text-align: justify">Ответственный за back-end и front-end разработку.
                        4 курс МИЭМ НИУ ВШЭ.</p>
                      <p class="subtitle is-6">@vladis2908</p>
                    </div>
                  </div>

                  <div class="content">
                     <a
                      class="is-flex is-flex-direction-row is-align-content-center mt-5"
                      href="https://vk.com/vladis2908"
                      target="_blank"
                      ><svg style="width:24px;height:24px" viewBox="0 0 24 24">
                        <path
                          fill="currentColor"
                          d="M15.07 2H8.93C3.33 2 2 3.33 2 8.93V15.07C2 20.67 3.33 22 8.93 22H15.07C20.67 22 22 20.67 22 15.07V8.93C22 3.33 20.67 2 15.07 2M18.15 16.27H16.69C16.14 16.27 15.97 15.82 15 14.83C14.12 14 13.74 13.88 13.53 13.88C13.24 13.88 13.15 13.96 13.15 14.38V15.69C13.15 16.04 13.04 16.26 12.11 16.26C10.57 16.26 8.86 15.32 7.66 13.59C5.85 11.05 5.36 9.13 5.36 8.75C5.36 8.54 5.43 8.34 5.85 8.34H7.32C7.69 8.34 7.83 8.5 7.97 8.9C8.69 11 9.89 12.8 10.38 12.8C10.57 12.8 10.65 12.71 10.65 12.25V10.1C10.6 9.12 10.07 9.03 10.07 8.68C10.07 8.5 10.21 8.34 10.44 8.34H12.73C13.04 8.34 13.15 8.5 13.15 8.88V11.77C13.15 12.08 13.28 12.19 13.38 12.19C13.56 12.19 13.72 12.08 14.05 11.74C15.1 10.57 15.85 8.76 15.85 8.76C15.95 8.55 16.11 8.35 16.5 8.35H17.93C18.37 8.35 18.47 8.58 18.37 8.89C18.19 9.74 16.41 12.25 16.43 12.25C16.27 12.5 16.21 12.61 16.43 12.9C16.58 13.11 17.09 13.55 17.43 13.94C18.05 14.65 18.53 15.24 18.66 15.65C18.77 16.06 18.57 16.27 18.15 16.27Z"
                        /></svg
                      ><span class="ml-3"></span>ВКонтакте</a
                    >
                    <a
                      class="is-flex is-flex-direction-row is-align-content-center mt-3"
                      href="https://github.com/vladisa88"
                      target="_blank"
                      ><svg style="width:24px;height:24px" viewBox="0 0 24 24">
                        <path
                          fill="currentColor"
                          d="M12,2A10,10 0 0,0 2,12C2,16.42 4.87,20.17 8.84,21.5C9.34,21.58 9.5,21.27 9.5,21C9.5,20.77 9.5,20.14 9.5,19.31C6.73,19.91 6.14,17.97 6.14,17.97C5.68,16.81 5.03,16.5 5.03,16.5C4.12,15.88 5.1,15.9 5.1,15.9C6.1,15.97 6.63,16.93 6.63,16.93C7.5,18.45 8.97,18 9.54,17.76C9.63,17.11 9.89,16.67 10.17,16.42C7.95,16.17 5.62,15.31 5.62,11.5C5.62,10.39 6,9.5 6.65,8.79C6.55,8.54 6.2,7.5 6.75,6.15C6.75,6.15 7.59,5.88 9.5,7.17C10.29,6.95 11.15,6.84 12,6.84C12.85,6.84 13.71,6.95 14.5,7.17C16.41,5.88 17.25,6.15 17.25,6.15C17.8,7.5 17.45,8.54 17.35,8.79C18,9.5 18.38,10.39 18.38,11.5C18.38,15.32 16.04,16.16 13.81,16.41C14.17,16.72 14.5,17.33 14.5,18.26C14.5,19.6 14.5,20.68 14.5,21C14.5,21.27 14.66,21.59 15.17,21.5C19.14,20.16 22,16.42 22,12A10,10 0 0,0 12,2Z"
                        /></svg
                      ><span class="ml-3"></span>Github</a
                    >
                    <!-- <a href="#">#css</a> <a href="#">#responsive</a> -->
                    <br>
                    <!-- <time datetime="2016-1-1">11:09 PM - 1 Jan 2016</time> -->
                  </div>
                </div>
              </div>
            </div>
            <div class="column is-half">
              <div class="card">
                <div class="card-image">
                  <figure class="image is-4by3">
                    <img src="./assets/images/plaxin.jpg">
                  </figure>
                </div>
                <div class="card-content">
                  <div class="media">
                    <div class="media-left">
                      <figure class="image is-48x48">
                        <img src="./assets/images/icon1.png" alt="Placeholder image">
                      </figure>
                    </div>
                    <div class="media-content">
                      <p class="title is-4" style="text-align: justify">Ответственный за разработку нейросети.
                        4 курс МИЭМ НИУ ВШЭ.</p>
                      <p class="subtitle is-6">@plxnvld</p>
                    </div>
                  </div>

                  <div class="content">
                     <a
                      class="is-flex is-flex-direction-row is-align-content-center mt-5"
                      href="https://vk.com/vladis2908"
                      target="_blank"
                      ><svg style="width:24px;height:24px" viewBox="0 0 24 24">
                        <path
                          fill="currentColor"
                          d="M15.07 2H8.93C3.33 2 2 3.33 2 8.93V15.07C2 20.67 3.33 22 8.93 22H15.07C20.67 22 22 20.67 22 15.07V8.93C22 3.33 20.67 2 15.07 2M18.15 16.27H16.69C16.14 16.27 15.97 15.82 15 14.83C14.12 14 13.74 13.88 13.53 13.88C13.24 13.88 13.15 13.96 13.15 14.38V15.69C13.15 16.04 13.04 16.26 12.11 16.26C10.57 16.26 8.86 15.32 7.66 13.59C5.85 11.05 5.36 9.13 5.36 8.75C5.36 8.54 5.43 8.34 5.85 8.34H7.32C7.69 8.34 7.83 8.5 7.97 8.9C8.69 11 9.89 12.8 10.38 12.8C10.57 12.8 10.65 12.71 10.65 12.25V10.1C10.6 9.12 10.07 9.03 10.07 8.68C10.07 8.5 10.21 8.34 10.44 8.34H12.73C13.04 8.34 13.15 8.5 13.15 8.88V11.77C13.15 12.08 13.28 12.19 13.38 12.19C13.56 12.19 13.72 12.08 14.05 11.74C15.1 10.57 15.85 8.76 15.85 8.76C15.95 8.55 16.11 8.35 16.5 8.35H17.93C18.37 8.35 18.47 8.58 18.37 8.89C18.19 9.74 16.41 12.25 16.43 12.25C16.27 12.5 16.21 12.61 16.43 12.9C16.58 13.11 17.09 13.55 17.43 13.94C18.05 14.65 18.53 15.24 18.66 15.65C18.77 16.06 18.57 16.27 18.15 16.27Z"
                        /></svg
                      ><span class="ml-3"></span>ВКонтакте</a
                    >
                    <a
                      class="is-flex is-flex-direction-row is-align-content-center mt-3"
                      href="https://github.com/vladisa88"
                      target="_blank"
                      ><svg style="width:24px;height:24px" viewBox="0 0 24 24">
                        <path
                          fill="currentColor"
                          d="M12,2A10,10 0 0,0 2,12C2,16.42 4.87,20.17 8.84,21.5C9.34,21.58 9.5,21.27 9.5,21C9.5,20.77 9.5,20.14 9.5,19.31C6.73,19.91 6.14,17.97 6.14,17.97C5.68,16.81 5.03,16.5 5.03,16.5C4.12,15.88 5.1,15.9 5.1,15.9C6.1,15.97 6.63,16.93 6.63,16.93C7.5,18.45 8.97,18 9.54,17.76C9.63,17.11 9.89,16.67 10.17,16.42C7.95,16.17 5.62,15.31 5.62,11.5C5.62,10.39 6,9.5 6.65,8.79C6.55,8.54 6.2,7.5 6.75,6.15C6.75,6.15 7.59,5.88 9.5,7.17C10.29,6.95 11.15,6.84 12,6.84C12.85,6.84 13.71,6.95 14.5,7.17C16.41,5.88 17.25,6.15 17.25,6.15C17.8,7.5 17.45,8.54 17.35,8.79C18,9.5 18.38,10.39 18.38,11.5C18.38,15.32 16.04,16.16 13.81,16.41C14.17,16.72 14.5,17.33 14.5,18.26C14.5,19.6 14.5,20.68 14.5,21C14.5,21.27 14.66,21.59 15.17,21.5C19.14,20.16 22,16.42 22,12A10,10 0 0,0 12,2Z"
                        /></svg
                      ><span class="ml-3"></span>Github</a
                    >
                    <!-- <a href="#">#css</a> <a href="#">#responsive</a> -->
                    <br>
                    <!-- <time datetime="2016-1-1">11:09 PM - 1 Jan 2016</time> -->
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section id="feedback-section" class="section columns">
          <div class="column is-full">
            <h1 class="section-title">
              Хотите с нами связаться? Отправьте обращение через форму ниже!
            </h1>
            <div>
              <input
                class="input is-large email-input"
                type="text"
                placeholder="Введите свой Email"
              />
              <br>
              <br>
              <textarea class="textarea" placeholder="Ваше обращение" rows="7"></textarea>
              <button class="send-button button is-large is-fullwidth">
                Отправить
              </button>
            </div>
          </div>
          <div class="column is-4 is-offset-2">
            <h1 class="section-title">
              Контакты
            </h1>
            <a
              class="is-flex is-flex-direction-row is-align-content-center mt-5"
              href="https://vk.com/vladis2908"
              target="_blank"
              ><svg style="width:24px;height:24px" viewBox="0 0 24 24">
                <path
                  fill="currentColor"
                  d="M15.07 2H8.93C3.33 2 2 3.33 2 8.93V15.07C2 20.67 3.33 22 8.93 22H15.07C20.67 22 22 20.67 22 15.07V8.93C22 3.33 20.67 2 15.07 2M18.15 16.27H16.69C16.14 16.27 15.97 15.82 15 14.83C14.12 14 13.74 13.88 13.53 13.88C13.24 13.88 13.15 13.96 13.15 14.38V15.69C13.15 16.04 13.04 16.26 12.11 16.26C10.57 16.26 8.86 15.32 7.66 13.59C5.85 11.05 5.36 9.13 5.36 8.75C5.36 8.54 5.43 8.34 5.85 8.34H7.32C7.69 8.34 7.83 8.5 7.97 8.9C8.69 11 9.89 12.8 10.38 12.8C10.57 12.8 10.65 12.71 10.65 12.25V10.1C10.6 9.12 10.07 9.03 10.07 8.68C10.07 8.5 10.21 8.34 10.44 8.34H12.73C13.04 8.34 13.15 8.5 13.15 8.88V11.77C13.15 12.08 13.28 12.19 13.38 12.19C13.56 12.19 13.72 12.08 14.05 11.74C15.1 10.57 15.85 8.76 15.85 8.76C15.95 8.55 16.11 8.35 16.5 8.35H17.93C18.37 8.35 18.47 8.58 18.37 8.89C18.19 9.74 16.41 12.25 16.43 12.25C16.27 12.5 16.21 12.61 16.43 12.9C16.58 13.11 17.09 13.55 17.43 13.94C18.05 14.65 18.53 15.24 18.66 15.65C18.77 16.06 18.57 16.27 18.15 16.27Z"
                /></svg
              ><span class="ml-3"></span>ВКонтакте</a
            >
            <a
              class="is-flex is-flex-direction-row is-align-content-center mt-3"
              href="https://facebook.com/askatsevalov"
              target="_blank"
              ><svg style="width:24px;height:24px" viewBox="0 0 24 24">
                <path
                  fill="currentColor"
                  d="M12 2.04C6.5 2.04 2 6.53 2 12.06C2 17.06 5.66 21.21 10.44 21.96V14.96H7.9V12.06H10.44V9.85C10.44 7.34 11.93 5.96 14.22 5.96C15.31 5.96 16.45 6.15 16.45 6.15V8.62H15.19C13.95 8.62 13.56 9.39 13.56 10.18V12.06H16.34L15.89 14.96H13.56V21.96A10 10 0 0 0 22 12.06C22 6.53 17.5 2.04 12 2.04Z"
                /></svg
              ><span class="ml-3"></span>Facebook</a
            >
            <a
              class="is-flex is-flex-direction-row is-align-content-center mt-3"
              href="https://github.com/vladisa88"
              target="_blank"
              ><svg style="width:24px;height:24px" viewBox="0 0 24 24">
                <path
                  fill="currentColor"
                  d="M12,2A10,10 0 0,0 2,12C2,16.42 4.87,20.17 8.84,21.5C9.34,21.58 9.5,21.27 9.5,21C9.5,20.77 9.5,20.14 9.5,19.31C6.73,19.91 6.14,17.97 6.14,17.97C5.68,16.81 5.03,16.5 5.03,16.5C4.12,15.88 5.1,15.9 5.1,15.9C6.1,15.97 6.63,16.93 6.63,16.93C7.5,18.45 8.97,18 9.54,17.76C9.63,17.11 9.89,16.67 10.17,16.42C7.95,16.17 5.62,15.31 5.62,11.5C5.62,10.39 6,9.5 6.65,8.79C6.55,8.54 6.2,7.5 6.75,6.15C6.75,6.15 7.59,5.88 9.5,7.17C10.29,6.95 11.15,6.84 12,6.84C12.85,6.84 13.71,6.95 14.5,7.17C16.41,5.88 17.25,6.15 17.25,6.15C17.8,7.5 17.45,8.54 17.35,8.79C18,9.5 18.38,10.39 18.38,11.5C18.38,15.32 16.04,16.16 13.81,16.41C14.17,16.72 14.5,17.33 14.5,18.26C14.5,19.6 14.5,20.68 14.5,21C14.5,21.27 14.66,21.59 15.17,21.5C19.14,20.16 22,16.42 22,12A10,10 0 0,0 12,2Z"
                /></svg
              ><span class="ml-3"></span>Github</a
            >
          </div>
        </section>
        <footer class="my-footer">
          <div class="content p-5 has-text-centered">
            <p>
              © HATEMETER | 2021 | Исаков Владислав и Плаксин Владислав
            </p>
          </div>
        </footer>
      </div>
    </div>

    <div v-if="step == 'load'" class="modal is-active is-clipped">
      <div class="modal-background"></div>
      <div class="load-modal modal-content is-clipped">
        <div class="box">
          <h1 class="title is-size-4">Вставьте ссылку на группу ВКонтакте</h1>
            <input class="input is-primary" v-model="url" placeholder="Ссылка, например, https://vk.com/groups/1234" >

          <button
            v-show="urlValidate"
            class="colorize-button button"
            @click="colorize"
          >
            <span class="text">Сделать фото цветным</span>
          </button>
        </div>
      </div>
      <button
        class="modal-close is-large"
        aria-label="close"
        @click="step = ''"
      ></button>
    </div>
    <div v-if="step == 'wait'" class="modal is-active is-clipped">
      <div class="modal-background"></div>
      <div class="wait-modal modal-content is-clipped">
        <div
          class="box is-flex is-flex-direction-column is-justify-content-center"
        >
          <span>Нейросеть обрабатывает фотографию, пожалуйста подождите</span>
          <progress class="progress is-small is-info mt-5"></progress>
        </div>
      </div>
      <button
        class="modal-close is-large"
        aria-label="close"
        @click="step = ''"
      ></button>
    </div>
    <div v-if="step == 'error'" class="modal is-active is-clipped">
      <div class="modal-background"></div>
      <div class="error-modal modal-content is-clipped">
        <div
          class="box is-flex is-flex-direction-column is-justify-content-center"
        >
          <span>Произошла ошибка, повторите попытку</span>
          <button class="button is-ghost mt-5" @click="colorize">
            Повторить
          </button>
        </div>
      </div>
      <button
        class="modal-close is-large"
        aria-label="close"
        @click="step = ''"
      ></button>
    </div>
    <div v-if="step == 'result'" class="modal is-active is-clipped">
      <div class="modal-background"></div>
      <div class="result-modal modal-content is-clipped">
        <div class="box">
          <h1 class="title is-size-4">Готово</h1>
          <figure class="image">
            <img class="result" :src="photoUrl" />
          </figure>
          <div
            class="is-flex is-flex-direction-row is-justify-content-space-between is-align-items-center mt-4"
          >
            <div class="field m-0">
              <label class="mr-2" for="switchRoundedOutlinedInfo">До</label>
              <input
                id="switchRoundedOutlinedInfo"
                type="checkbox"
                name="switchRoundedOutlinedInfo"
                class="switch is-rounded is-info"
                v-model="showOutput"
              />
              <label for="switchRoundedOutlinedInfo">После</label>
            </div>
            <a
              class="button is-ghost"
              :download="outputPhoto.name"
              :href="`/storage/${outputPhoto.name}`"
            >
              Скачать результат
            </a>
          </div>
        </div>
      </div>
      <button
        class="modal-close is-large"
        aria-label="close"
        @click="step = ''"
      ></button>
    </div>

    <div
      v-show="notificationVisibility"
      class="my-notification notification is-danger"
    >
      <button class="delete" @click="notificationVisibility = false"></button>
      {{ notificationMessage }}
    </div>
  </div>
</template>

<script>
import { Carousel, Slide } from 'vue-carousel';
import utils from './utils'

export default {
  name: "App",
  components: {
    Carousel,
    Slide
  },
  data() {
    return {
      step: "",
      url: "",
      aboutUsData: 
      [
          {
            photo: './assets/images/isakov.jpg',
            name: 'Исаков Владислав',
            description: '',
            link: 'https://github.com/vladisa88',
            nickname: '@vladis2908'
          },
          {
            photo: './assets/images/plaxin.jpg',
            name: 'Плаксин Владислав',
            description: '',
            link: '@plxnvld',
            nickname: '@plxnvld'
          },

      ],
      notificationVisibility: false,
      modalTitle: "",
      notificationMessage: "",
      inputPhoto: {
        name: "",
      },
      outputPhoto: {
        name: "",
      },
      showOutput: true,
      showNavbarShadow: false,
    };
  },
  computed: {
    photoUrl() {
      return `/storage/${
        this.showOutput ? this.outputPhoto.name : this.inputPhoto.name
      }`;
    },
    urlValidate() {
      return true ? this.url.indexOf('https://vk.com/groups/') != -1 : false
    },
    test() {
      console.log(this.aboutUsData)
    }
  },
  methods: {
    onScroll() {
      const currentScrollPosition = document.getElementsByClassName("main")[0]
        .scrollTop;
      if (currentScrollPosition < 0) {
        return;
      }
      this.showNavbarShadow = currentScrollPosition > 0;
    },
    start() {
      this.readyToColorize = false;
      this.step = "load";
    },
    async colorize() {
      this.step = "wait";
      this.showOutput = true;
      var response = await this.axios.post(
        `/api/colorizer/${this.inputPhoto.id}`
      );
      if (response.status == 200) {
        this.outputPhoto = response.data;
        this.step = "result";
      } else {
        this.outputPhoto = null;
        this.step = "error";
      }
    },
    fileUploaded(file, response) {
      this.inputPhoto = response;
      this.readyToColorize = true;
    },
    removeBadPhoto(file) {
      this.notificationMessage =
        "Загруженное фото не подходит! Попробуйте другое...";
      this.notificationVisibility = true;
      this.$refs.myDropzone.removeFile(file);
    },
    removeExtraPhoto(file) {
      this.notificationMessage = "Можно загружать только одно фото за раз!";
      this.notificationVisibility = true;
      this.$refs.myDropzone.removeFile(file);
    },
  },
};
</script>

<style lang="scss">
@font-face {
  font-family: "Didot";
  font-weight: 400;
  src: local("Gilroy"),
    url(./assets/fonts/Gilroy-Regular.ttf) format("truetype");
}

@font-face {
  font-family: "Helvetica";
  font-weight: 900;
  src: local("Helvetica"),
    url(./assets/fonts/Helvetica-Bold.ttf) format("truetype");
}
html {
  overflow-y: hidden !important;
  & * {
    font-family: "Helvetica" !important;
  }
}
.bottom-border {
  box-shadow: 0px 0px 4px 1px darkgray;
  transition: 0.2s;
}
.dark-link {
  color: #1f2c58 !important;
  font-size: 22px !important;
  line-height: 20px !important;
}
.my-container {
  @media screen and (min-width: 768px) {
    padding-right: 80px;
    padding-left: 80px;
  }
  overflow-y: auto;
  scroll-behavior: smooth;
  color: #1f2c58;
}
.main-title {
  font-style: normal;
  font-weight: 900;
  font-size: 100px;
  line-height: 100px;
  width: 100%;
}
.start-button {
  margin-top: 80px;
  background: linear-gradient(90deg, #70c58a 0%, #66bb6d 50%, #2bb11f 100%);
  border-radius: 2px;
  width: 305px;
  height: 60px !important;

  .text {
    text-align: center;
    color: #ffffff;
    font-style: normal;
    font-weight: 600;
    font-size: 16px;
    line-height: 20px;
  }
}
.colorize-button {
  margin-top: 20px;
  background: linear-gradient(90deg, #3981aa 0%, #3966aa 50%, #3949aa 100%);
  border-radius: 2px;
  width: 100%;
  height: 60px !important;

  .text {
    text-align: center;
    color: #ffffff;
    font-style: normal;
    font-weight: 600;
    font-size: 16px;
    line-height: 20px;
  }
}
.load-modal {
  max-width: 400px;
}
.wait-modal {
  max-width: 600px;
  .box {
    height: 280px;
    padding-left: 150px;
    padding-right: 150px;
    text-align: center;
  }
}
.error-modal {
  max-width: 600px;
  .box {
    height: 280px;
    padding-left: 100px;
    padding-right: 100px;
    text-align: center;
  }
}
.footer {
  position: absolute;
  bottom: 0;
  right: 0;
  left: 0;
  padding: 1rem !important;
  background-color: rgba($color: #eeeeee, $alpha: 0.8);
}
#dropzone {
  border: 2px dashed #b9c2eb;
  box-sizing: border-box;
  border-radius: 2px;
  height: 300px;
}
.my-notification {
  position: absolute !important;
  left: 5%;
  top: 5%;
  width: 30vw;
  z-index: 1000;
}
.result {
  max-height: 80vh;
}
.navbar {
  height: 120px;
}
#main-section {
  margin-top: 50px;
}
#about-nn-section {
  margin-top: 150px;
}
.logo {
  width: 120px;
  height: 100%;
}
.main {
  overflow-y: auto;
  overflow-x: hidden;
  height: 100%;
  flex: 1;
}

.main::-webkit-scrollbar {
  width: 0.25rem;
}

.main::-webkit-scrollbar-track {
  background: ghostwhite;
}

.main::-webkit-scrollbar-thumb {
  background: hsl(0, 0%, 29%);
}
.fullheight {
  height: calc(100vh - 120px);
  margin-top: 120px;
  // background-color: #FFB3A5;
}
.section-title {
  font-size: 40px;
  font-weight: 900;
}
.email-input {
  margin-top: 40px;
  // background-color: #eeeeee !important;
  // border: none !important;
}
.send-button {
  margin-top: 24px;
  background-color: #3981aa !important;
  color: #fff !important;
}

.VueCarousel-slide {
  position: relative;
  background: #eee;
  color: #fff;
  font-family: Arial;
  font-size: 24px;
  text-align: center;
  height: 100%;
}

.label {
  position: absolute;
  top: 70%;
  left: 50%;
  transform: translate(-50%, -50%);
}

div.card {
  width: 580px;
  margin: 1rem;
}

// div.main-back {
//   background-image: url('./assets/images/background.png');
// }
</style>
