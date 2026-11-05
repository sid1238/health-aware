import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import App from './App';

jest.mock('axios', () => ({ __esModule: true, default: { post: jest.fn() } }));

const goTo = (path) => window.history.pushState({}, '', path);

beforeEach(() => {
  window.scrollTo = jest.fn();
  Element.prototype.scrollIntoView = jest.fn();
  axios.post.mockReset();
});

test('home page shows the conditions and myths sections', () => {
  goTo('/');
  render(<App />);
  expect(screen.getByRole('heading', { name: /most prevalent health conditions/i })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: /myths about mental health/i })).toBeInTheDocument();
});

test('purpose page shows mission, vision and values', () => {
  goTo('/purpose');
  render(<App />);
  ['Our Mission', 'Our Vision', 'Our Values'].forEach((name) => {
    expect(screen.getByRole('heading', { name })).toBeInTheDocument();
  });
});

test('help page shows the answer with linked sources', async () => {
  axios.post.mockResolvedValue({
    data: {
      answer: 'Stomach upset is the most common side effect.',
      sources: [{ title: 'Ibuprofen', url: 'https://en.wikipedia.org/wiki/Ibuprofen' }],
    },
  });
  goTo('/help');
  render(<App />);

  fireEvent.change(screen.getByLabelText(/your health question/i), {
    target: { value: 'What are the side effects of ibuprofen?' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

  expect(await screen.findByText(/stomach upset/i)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Ibuprofen' })).toHaveAttribute(
    'href',
    'https://en.wikipedia.org/wiki/Ibuprofen'
  );
});

test('help page explains when the backend is unreachable', async () => {
  axios.post.mockRejectedValue(new Error('Network Error'));
  goTo('/help');
  render(<App />);

  fireEvent.click(screen.getByRole('button', { name: /side effects of ibuprofen/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent(/couldn't reach the healthbot server/i);
});

test('a question linked from the home page is asked exactly once', async () => {
  axios.post.mockResolvedValue({ data: { answer: 'Linked answer.', sources: [] } });
  goTo('/help?q=' + encodeURIComponent('Are carbs bad?'));
  render(<App />);

  expect(await screen.findByText('Linked answer.')).toBeInTheDocument();
  await waitFor(() => expect(window.location.search).toBe(''));
  expect(axios.post).toHaveBeenCalledTimes(1);
  expect(axios.post.mock.calls[0][1]).toEqual({ question: 'Are carbs bad?' });
});
